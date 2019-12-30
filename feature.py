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
        np.savetxt("C:\\Users\\Cecilia\\Desktop\\keys_en.txt", mylsa.keys, fmt="%s")
        references = doc_matrix[: ref_num]  # terms in references are Class Sentence and are preprocessed already.

        for i in range(ref_num, len(doc_matrix)):
            if record_count == 460:
                break
            current_answer = doc_matrix[i]
            textid = textids[i]
            print("This is NO.", i - ref_num + 1, "text with textid--", textid)

            # Calculate features including LENGTHRATIO, 1~4GRAM, LSAGRADE, VEC_SIM, etc.
            lengthratio = get_lengthratio(refs=references, answer=current_answer)
            bleu = get_bleu_score(refs=references, answer=current_answer)
            lsagrade = mylsa.get_max_similarity(10, i, ref_num)
            vec_sim = get_min_WCD(id=i, vecs=word2vec_model, stopwords=[], tf_idf=mylsa.A,
                                  keys=mylsa.keys, ref_num=ref_num, language=language)
            # vec_sim = get_min_WMD(refs=references, sentence=current_answer, vectors=word2vec_model)
            fluency = get_fluency_score(refs=references, sentence=current_answer)
            np_length_ratio = get_phrases_ratio(phrase="NP", refs=references, answer=current_answer, model=nlp_model)
            vp_length_ratio = get_phrases_ratio(phrase="VP", refs=references, answer=current_answer, model=nlp_model)

            # Insert features of a certain text into DB
            if insert_features(course=courseid, conn=conn, textid=textid, ngram=bleu, lengthratio=lengthratio,
                               lsagrade=lsagrade, vec_sim=vec_sim, fluency=fluency,
                               np=np_length_ratio, vp=vp_length_ratio):
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


def insert_features(course, conn, textid, ngram, lengthratio, lsagrade, vec_sim, fluency, np, vp):
    """
    Insert a feature record into DB.
    Parameters:
        course: String, courseid
        conn: A mysql connection.
        textid, ngram, lengthratio, lsagrade, vec_sim: Features to insert into DB
    Returns:
        Boolean, true if success inserting else false.
    """

    try:
        insert_feature_sql = "INSERT INTO features_" + course +\
                             "(textid, 1gram, 2gram, 3gram, 4gram, lengthratio, lsagrade, vecsim, fluency, np, vp)" \
                             "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        insert_feature_cur = conn.cursor()
        print("Inserting into features_" + course + " -- textid:", textid, "1~4gram:", ngram, "lengthratio:", lengthratio,
              "lsa:", lsagrade, "vec:", vec_sim, "fluency:", fluency, "np:", np, "vp:", vp)
        insert_feature_cur.execute(insert_feature_sql, (
            textid, ngram[0], ngram[1], ngram[2], ngram[3], lengthratio, lsagrade, vec_sim, fluency, np, vp))
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
    for ref in refs:
        ref_ngram = ref.ngram
        answer_ngram = answer.ngram
        total_count = [0] * 4
        match_count = [0] * 4
        for key in answer_ngram.keys():
            n = len(key.split(" ")) - 1   # key is (n+1)gram
            total_count[n] += 1
            if key in ref_ngram.keys():
                match_count[n] += min(answer_ngram[key], ref_ngram[key])
                # print("Got one:", key, "+", min(answer_ngram[key], ref_ngram[key]))
        # bleu formula.
        # score = math.exp(sum([math.log(float(a)/b) for a, b in zip(match_count, total_count)]) * 0.25)
        for i in range(4):
            score_list[i] = max(float(score_list[i]), float(match_count[i]) / total_count[i])
        # score_list.append(score)
    return score_list


def extract_data(conn, course):
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
    get_all_features_sql = "SELECT textid, 1gram, 2gram, 3gram, 4gram, lengthratio, vecsim" \
                           " FROM features_" + course
    get_questionid_sql = "SELECT questionid FROM detection WHERE textid=%s"
    cur = conn.cursor()
    feature_list = []
    score_list = []
    score_text = {}  # key is textid and value is score(aka. y) of the text
    try:
        cur.execute(get_all_features_sql)
        features = cur.fetchall()
        # Get feature(aka. X) values
        for f in features:
            feature_list.append(list(f[1:]))
        features = np.asarray(features)
        # Get the matching score for every text
        for text_id in features[:, 0]:

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
        conn.close()


def cor_of_features(features, scores):
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
    features_arr = np.asarray(features)
    for f in ['1gram', '2gram', '3gram', '4gram', 'lengthratio', 'lsagrade', 'vec', 'fluency', 'np', 'vp']:
        cors[f] = round(pearson_cor(scores, features_arr[:, i]), 4)
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

    # word_vectors_en = KeyedVectors.load("./model/vectors_en.kv")
    # en_model = StanfordCoreNLP(r"H:\Download\stanford-corenlp-full-2018-02-27")
    # cal_features(conn=conn, courseid="202英语二", word2vec_model=word_vectors_en, nlp_model=en_model)
    feature, score = extract_data(conn=conn, course="202英语二")

    # cor_of_features(feature, score)



