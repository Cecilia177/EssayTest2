import pymysql
from sentence import Sentence
from LSA import LSA
import traceback
import numpy as np
from correlation import pearson_cor
from vecsim import get_min_WCD, get_min_WMD
from gensim.models import KeyedVectors
from fluency import get_fluency_score
from stanfordcorenlp import StanfordCoreNLP
from partofspeech import get_phrases_ratio
import math

pymysql.converters.encoders[np.float64] = pymysql.converters.escape_float
pymysql.converters.conversions = pymysql.converters.encoders.copy()
pymysql.converters.conversions.update(pymysql.converters.decoders)


def cal_features(conn, courseid, word2vec_model, nlp_model):
    """
    Calculate features of texts of all courses and questions and insert those into DB.
    To put it simple, complete the features table in DB.
    Parameters:
        conn: A mysql connection.

    """
    language = "ch" if courseid == "201英语一" else "en"
    # Get all questions of the certain course.
    try:
        get_cour_sql = "SELECT DISTINCT questionid FROM standards where courseid=%s"
        get_cour_cur = conn.cursor()
        get_cour_cur.execute(get_cour_sql, courseid)
        ques = get_cour_cur.fetchall()
    except Exception as e:
        print("Error getting courses and questions!", e)
    finally:
        get_cour_cur.close()

    # Truncating feature table before inserting.
    delete_feature_record(conn, courseid)

    record_count = 0
    for questionid in ques:
        print("--------------------current question:", questionid, "of", courseid, "----------------------")

        # Build docs matrix for every question.
        if get_docs_list(conn=conn, courseid=courseid, questionid=questionid) is None:
            continue
        ref_num, textids, doc_matrix = get_docs_list(conn=conn, courseid=courseid, questionid=questionid)
        mylsa = build_svd(doc_matrix)
        # np.savetxt("C:\\Users\\Cecilia\\Desktop\\keys_en.txt", mylsa.keys, fmt="%s")
        references = doc_matrix[: ref_num]  # terms in references are Class Sentence and are preprocessed already.

        for i in range(ref_num, len(doc_matrix)):
            # if record_count == 460:
            #     break
            current_answer = doc_matrix[i]
            textid = textids[i]
            print("This is NO.", i - ref_num + 1, "text with textid--", textid)
            features = {}
            features['textid'] = textid

            # Calculate features including LENGTHRATIO, 1~4GRAM, LSAGRADE, VEC_SIM, etc.
            lengthratio = get_lengthratio(refs=references, answer=current_answer)
            ngrams, bleu = get_bleu_score(refs=references, answer=current_answer)
            lsagrade = mylsa.get_max_similarity(10, i, ref_num)
            vec_sim = get_min_WCD(id=i, vecs=word2vec_model, stopwords=[], tf_idf=mylsa.A,
                                  keys=mylsa.keys, ref_num=ref_num, language=language)
            # vec_sim = get_min_WMD(refs=references, sentence=current_answer, vectors=word2vec_model)
            fluency = get_fluency_score(refs=references, sentence=current_answer)
            np_length_ratio = get_phrases_ratio(phrase="NP", refs=references, answer=current_answer, model=nlp_model)
            vp_length_ratio = get_phrases_ratio(phrase="VP", refs=references, answer=current_answer, model=nlp_model)
            features['1gram'] = ngrams[0]
            features['2gram'] = ngrams[1]
            features['3gram'] = ngrams[2]
            features['4gram'] = ngrams[3]
            features['bleu'] = bleu
            features['lengthratio'] = lengthratio
            features['lsagrade'] = lsagrade
            features['vecsim'] = vec_sim
            features['fluency'] = fluency
            features['np'] = np_length_ratio
            features['vp'] = vp_length_ratio
            # Insert features of a certain text into DB
            if insert_features(course=courseid, conn=conn, features=features):
                record_count += 1
    print("--------------------Finishing inserting features of", record_count, "text.----------------------")


def delete_feature_record(conn, course):
    """
    Deleting all records of feature table.
    Parameters:
        conn: A mysql connection.
        course: String, courseid
    """
    try:
        truncate_sql = "TRUNCATE features_" + course
        truncate_cur = conn.cursor()
        truncate_cur.execute(truncate_sql)
    except Exception as e:
        print("Error truncating table..", traceback.print_exc())
        conn.rollback()
    finally:
        truncate_cur.close()


def insert_features(course, conn, features):
    """
    Insert a feature record into DB.
    Parameters:
        course: String, courseid
        conn: A mysql connection.
        features: A dict, keys ranges in 'textid', '1gram', '2gram', '3gram', '4gram', 'lengthratio', 'lsagrade',
            'vecsim', 'fluency', 'np', 'vp'
    Returns:
        Boolean, true if success inserting else false.
    """
    if "textid" not in features.keys():
        print("FAILURE: NO textid included in features!")
        return False
    flag = False
    try:
        select_feature_sql = "SELECT * FROM features_" + course + "  WHERE textid=%s"
        select_cur = conn.cursor()
        select_cur.execute(select_feature_sql, features['textid'])
        if select_cur.fetchone() is not None:
            flag = True
    except Exception:
        print("Error selecting features of textid ", features['textid'], traceback.print_exc())
    finally:
        select_cur.close()

    # if textid is already in db, then update instead of inserting.
    try:
        features_to_insert = ", ".join(features.keys())
        insert_feature_sql = "INSERT INTO features_" + course + " (" + features_to_insert + ") " + \
                             "VALUES (%(" + ")s, %(".join(features.keys()) + ")s)"
        update_feature_sql = "UPDATE features_" + course + " SET "
        for f in features.keys():
            if f == 'textid':
                continue
            update_feature_sql += f + "=%(" + f+")s, "
        update_feature_sql = update_feature_sql[:-2] + " WHERE textid=%(textid)s"
        sql = update_feature_sql if flag else insert_feature_sql
        # print("final sql:", sql)
        insert_feature_cur = conn.cursor()
        insert_feature_cur.execute(sql, features)
        conn.commit()
        print("Success!")
        return True
    except Exception as e:
        print("Error inserting features..", traceback.print_exc())
        conn.rollback()
        return False
    finally:
        insert_feature_cur.close()


def build_svd(docs_list):
    """
    Parameters:
        A list of docs.
    Returns:
        A LSA object, containing matrix A as tf-idf matrix,
            and method get_similarity() to cal the similarity between 2 docs in docs_list.
    """
    # build count matrix, tf-idf modification matrix and get svd.
    lsa = LSA(stopwords=[], ignorechars="")
    for doc in docs_list:
        lsa.parse(doc)
    lsa.build_count_matrix()
    lsa.TFIDF()
    lsa.svd_cal()
    return lsa


def get_docs_list(conn, courseid, questionid):
    """
    Parameters:
        conn: A mysql connection.
        courseid: String
        questionid: Integer
    Return:
        Integer: number of references
        Two lists: list of textid and list of doc(class Sentence), the first terms of both are -1 and Sentence reference
    """
    # Get reference of certain courseid and questionid.
    try:
        get_ref_sql = "SELECT ref FROM standards WHERE courseid=%s AND questionid=%s"
        get_ref_cur = conn.cursor()
        get_ref_cur.execute(get_ref_sql, (courseid, questionid))
        refs = get_ref_cur.fetchall()
        # ref = get_ref_cur.fetchone()[0]
    except Exception as e:
        print("Error getting reference of courseid", courseid, "questionid", questionid)
        print(traceback.print_exc())
    finally:
        get_ref_cur.close()

    lang = "ch" if courseid == "201英语一" else "en"
    doc_matrix = []
    textids = []
    ref_id = -1
    for ref in refs:
        reference = Sentence(text=ref[0], language=lang)
        reference.preprocess()
        doc_matrix.append(reference)   # add Sentence reference as the first term of doc_matrix
        textids.append(ref_id)   # Use negative numbers as reference textid.
        ref_id -= 1
    # Get all detection text of certain courseid and questionid.
    detections = None
    try:
        get_detection_sql = "SELECT textid, text FROM detection WHERE courseid = %s and questionid = %s"
        get_detection_cur = conn.cursor()
        if get_detection_cur.execute(get_detection_sql, (courseid, questionid)):
            detections = get_detection_cur.fetchall()
        else:
            print("No quesion", questionid, "of", courseid, "in DETECTION DB.")
    except Exception as e:
        print("Error getting text...", traceback.print_exc())
    finally:
        get_detection_cur.close()

    # Add all detections into doc_matrix
    if detections is None:
        return
    for dt in detections:
        textids.append(dt[0])
        cur_ans = Sentence(text=dt[1], language=lang)
        cur_ans.preprocess()
        doc_matrix.append(cur_ans)
    return len(refs), textids, doc_matrix


def get_lengthratio(refs, answer):
    """
    Parameters:
        refs: List, item of which is class Sentence
        answer: class Sentence
    Return:
        float, answer length / average ref length
    """
    ref_length = 0
    for ref in refs:
        ref_length += ref.seg_length
    length = float(ref_length) / len(refs)
    return answer.seg_length / length


def get_bleu_score(refs, answer):
    """
    paras:
        refs: List, item of which is class Sentence
        answer: class Sentence
    Return:
        A list including maximum 1~4gram matching rate of answer compared to all the refs,
            eg. [1gram rate, 2gram rate, 3gram rate, 4gram rate]
    """
    score_list = [0] * 4
    bleu = 0
    for ref in refs:
        ref_ngram = ref.ngram
        answer_ngram = answer.ngram
        total_count = [0] * 4
        match_count = [0] * 4
        addone_total_count = [0] * 4
        for key in answer_ngram.keys():
            n = len(key.split(" ")) - 1   # key is (n+1)gram
            total_count[n] += answer_ngram[key]
            addone_total_count[n] += answer_ngram[key] + 1   # add-one-smoothing
            if key in ref_ngram.keys():
                match_count[n] += min(answer_ngram[key], ref_ngram[key])
                # print("Got one:", key, "+", min(answer_ngram[key], ref_ngram[key]))
        # bleu formula.
        addone_match_count = [c+1 for c in match_count]   # add-one-smoothing
        # punishing for being too short
        bp = 1 if answer.seg_length > ref.seg_length else math.exp(1 - ref.seg_length / answer.seg_length)
        bleu_score = bp * math.exp(sum([math.log(float(a)/b) for a, b in zip(addone_match_count, addone_total_count)]) * 0.25)
        bleu = max(bleu, bleu_score)
        for i in range(4):
            score_list[i] = max(float(score_list[i]), float(match_count[i]) / total_count[i])
    return score_list, bleu


def extract_data(conn, course, features):
    """
    Get the correlation of score(y) and each feature.
    Parameters:
        conn: A mysql connection
        course: String, courseid
    Return:
        feature_list: the matrix of feature values, shape of which is (M, N).
                --M is the number of samples and N is the number of features(6 for now).
                --features[i][j] is the NO.j feature value of NO.i sample.
            feature sequence is ['1gram', '2gram', '3gram', '4gram', 'lengthratio', 'lsagrade'].
        score_list:
            matrix of scores, shape of which is (M, 1).
                --M is the number of samples.
    """
    get_all_features_sql = "SELECT textid, " + ", ".join(features) + " FROM features_" + course
    get_questionid_sql = "SELECT questionid FROM detection WHERE textid=%s"
    cur = conn.cursor()
    feature_list = []
    score_list = []
    score_text = {}  # key is textid and value is score(aka. y) of the text
    try:
        cur.execute(get_all_features_sql)
        features_data = cur.fetchall()
        # Get feature(aka. X) values
        for f in features_data:
            feature_list.append(list(f[1:]))
        features_data = np.asarray(features_data)
        # Get the matching score for every text
        for text_id in features_data[:, 0]:
            cur.execute(get_questionid_sql, text_id)
            question_id = cur.fetchone()[0]
            get_score_sql = "SELECT z" + str(question_id) + " FROM scores, detection WHERE detection.textid=%s " \
                            "and scores.studentid=detection.studentid"
            cur.execute(get_score_sql, text_id)
            score = cur.fetchone()[0]
            score_list.append(score)
            score_text[text_id] = score
        return feature_list, score_list
    except Exception:
        print("Error getting features...", traceback.print_exc())
    finally:
        cur.close()
        # conn.close()


def cor_of_features(conn, courseid, features):
    """
    Paras:
        features:
            the matrix of feature values, shape of which is (M, N).
                --M is the number of samples and N is the number of features(8 for now).
                --features[i][j] is the NO.j feature value of NO.i sample.
            feature sequence is ['1gram', '2gram', '3gram', '4gram', 'lengthratio', 'lsagrade', 'vecsim', 'fluency'].
        scores:
            matrix of scores, shape of which is (M, 1).
                --M is the number of samples.
    returns:
        A dict, key of which is feature name and value is pearson correlation value.
    """
    cors = {}
    i = 0
    feature_list, score_list = extract_data(conn=conn, course=courseid, features=features)
    features_arr = np.asarray(feature_list)
    for f in features:
        cors[f] = round(pearson_cor(score_list, features_arr[:, i]), 4)
        i += 1
    print(cors)
    return cors


if __name__ == '__main__':
    conn = pymysql.connect(host="127.0.0.1",
                           database='essaydata',
                           port=3306,
                           user='root',
                           password='',
                           charset='utf8')

    # word_vectors = KeyedVectors.load("./model/vectors.kv")
    # zh_model = StanfordCoreNLP(r"H:\Download\stanford-corenlp-full-2018-02-27", lang='zh')
    # cal_features(conn=conn, courseid="201英语一", word2vec_model=word_vectors, nlp_model=zh_model)
    # feature, score = extract_data(conn=conn, course="201英语一")

    word_vectors_en = KeyedVectors.load("./model/vectors_en.kv")
    en_model = StanfordCoreNLP(r"H:\Download\stanford-corenlp-full-2018-02-27")

    cal_features(conn=conn, courseid="202英语二", word2vec_model=word_vectors_en, nlp_model=en_model)
    # feature, score = extract_data(conn=conn, course="202英语二")
    features = ['1gram', '2gram', '3gram', '4gram', 'lengthratio', 'lsagrade', 'vecsim', 'fluency', 'np', 'vp', 'bleu']
    cor_of_features(conn=conn, courseid="202英语二", features=features)



