import pandas as pd
import pymysql
import traceback
import numpy as np
import random

def pearson_cor(y_true, y_predict):
    """
    Calculate the Pearson correlation coefficient between a and b
    :param y_true:
    :param y_predict: both y_true and y_predict are Array and their lengths are the same.
    :return: Pearson correlation coefficient value.
    """
    if type(y_true[0]) != float:
        y_true = [float(s) for s in y_true]
    if type(y_predict[0]) != float:
        y_predict = [float(s) for s in y_predict]
    y1 = pd.Series(y_true)
    y2 = pd.Series(y_predict)
    return y1.corr(y2, method='pearson')


def spearman_cor(y_true, y_predict):
    """
    The same like the above function pearson except that it's Spearman correlation coefficient value here.
    """
    if type(y_true) == str:
        y_true = [float(s) for s in y_true]
    if type(y_predict) == str:
        y_predict = [float(s) for s in y_predict]
    y1 = pd.Series(y_true)
    y2 = pd.Series(y_predict)
    return y1.corr(y2, method='spearman')


if __name__ == '__main__':
    # get correlation of two teachers' correlation
    conn = pymysql.connect(host="127.0.0.1",
                           database='essaydata',
                           port=3306,
                           user='root',
                           password='',
                           charset='utf8')
    sql = "SELECT scoreof_1, scoreof_2 FROM detailed_score AA, detection BB WHERE BB.courseid=%s AND BB.questionid=%s " \
          "AND BB.studentid = AA.studentid AND AA.questionid=BB.questionid AND AA.courseid=BB.courseid"
    sql1 = "SELECT studentid, scoreof_1, scoreof_2 FROM detailed_score WHERE courseid=%s AND questionid=%s"
    sql2 = "SELECT studentid FROM detection WHERE courseid=%s AND questionid=%s"
    sql3 = "SELECT studentid FROM detailed_score WHERE courseid=%s AND questionid=%s"
    cur = conn.cursor()
    all_scores = None
    todelete = random.sample(range(0, 150), 38)
    try:
        cur.execute(sql2, ("201英语一", 1))
        detection = np.asarray(cur.fetchall())[:, 0]
        cur.execute(sql3, ("201英语一", 1))
        scores_list = np.asarray(cur.fetchall()[:150])[:, 0]
        count = 0
        for d in scores_list:
            if d not in detection:
                count += 1
        print("delete count:", count)

        for questionid in range(1, 2):
            cur.execute(sql1, ("201英语一", questionid))
            # cur.execute(sql1)
            scores = np.asarray(cur.fetchall()[:150])
            new_scores = []
            for s in scores:
                if s[0] in detection:
                    print(s)
                    new_scores.append(s)
            print(questionid, ":", len(new_scores))
            for k in range(0, 22):
                new_scores.append(['1111', '0.0', '0.0'])
            new_scores = np.asarray(new_scores)
            print(new_scores.shape)
            print(pearson_cor(new_scores[:, 1], new_scores[:, 2]))

            all_scores = scores if all_scores is None else np.vstack((all_scores, scores))
        print(all_scores.shape)
    except Exception as e:
        print("Error getting scores...", traceback.print_exc())
    finally:
        cur.close()
        conn.close()
    scores1 = all_scores[:, 1]
    scores2 = all_scores[:, 2]
    print(pearson_cor(scores1, scores2))

