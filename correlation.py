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
    The same as the above function pearson_cor except that it's Spearman correlation coefficient value here.
    """
    if type(y_true) == str:
        y_true = [float(s) for s in y_true]
    if type(y_predict) == str:
        y_predict = [float(s) for s in y_predict]
    y1 = pd.Series(y_true)
    y2 = pd.Series(y_predict)
    return y1.corr(y2, method='spearman')


def get_correlation_of_grades(course, questionids, col_type):
    """
    Calculate correlation of two graders of a few questions in a course.
    Parameters:
        course: courseid
        questionids: A list of questionid
        col_type: A correlation calculation function, callable.
    Return:
        The col_type value.
    """
    conn = pymysql.connect(host="127.0.0.1",
                           database='essaydata',
                           port=3306,
                           user='root',
                           password='',
                           charset='utf8')
    sql = "SELECT scoreof_1, scoreof_2 FROM detailed_score AA, detection_copy1 BB WHERE BB.courseid=%s AND BB.questionid=%s " \
          "AND BB.studentid = AA.studentid AND AA.questionid=BB.questionid AND AA.courseid=BB.courseid "
    cur = conn.cursor()
    all_scores = None
    try:
        for questionid in questionids:
            cur.execute(sql, (course, questionid))
            scores = np.asarray(cur.fetchall())
            all_scores = scores if all_scores is None else np.vstack((all_scores, scores))
        print(all_scores.shape)
    except Exception as e:
        print("Error getting scores...", traceback.print_exc())
    finally:
        cur.close()
        conn.close()
    scores1 = all_scores[:, 0]
    scores2 = all_scores[:, 1]
    return col_type(scores1, scores2)


if __name__ == '__main__':
    # get correlation of two teachers' correlation
    print(get_correlation_of_grades("201英语一", range(1, 6), pearson_cor))


