import streamlit as st

import numpy as np
import pandas as pd

import sqlite3

conn = sqlite3.connect('student_feedback.db')
c = conn.cursor()


def create_table():
    c.execute(
        'CREATE TABLE IF NOT EXISTS feedback(date_submitted DATE, Q1 TEXT, Q2 INTEGER, Q3 INTEGER, Q4 TEXT, Q5 TEXT, Q6 TEXT, Q7 TEXT, Q8 TEXT)')


def add_feedback(date_submitted, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8):
    c.execute('INSERT INTO feedback (date_submitted,Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8) VALUES (?,?,?,?,?,?,?,?,?)',
              (date_submitted, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8))
    conn.commit()


def main():
    st.title("Student Feedback")

    d = st.date_input("Today's date", None, None, None, None)

    question_1 = st.selectbox('Who was your teacher?',
                              ('', 'Mr Thomson', 'Mr Tang', 'Ms Taylor', 'Ms Rivas', 'Mr Hindle', 'Mr Henderson'))
    st.write('You selected:', question_1)

    question_2 = st.slider('What year are you in?', 7, 13)
    st.write('You selected:', question_2)

    question_3 = st.slider(
        'Overall, how happy are you with the lesson? (5 being very happy and 1 being very dissapointed)', 1, 5, 1)
    st.write('You selected:', question_3)

    question_4 = st.selectbox('Was the lesson fun and interactive?', ('', 'Yes', 'No'))
    st.write('You selected:', question_4)

    question_5 = st.selectbox('Was the lesson interesting and engaging?', ('', 'Yes', 'No'))
    st.write('You selected:', question_5)

    question_6 = st.selectbox('Were you content with the pace of the lesson?', ('', 'Yes', 'No'))
    st.write('You selected:', question_6)

    question_7 = st.selectbox('Did your teacher explore the real-world applications of what you learnt?',
                              ('', 'Yes', 'No'))
    st.write('You selected:', question_7)

    question_8 = st.text_input('What could have been better?', max_chars=50)

    if st.button("Submit feedback"):
        create_table()
        add_feedback(d, question_1, question_2, question_3, question_4, question_5, question_6, question_7, question_8)
        st.success("Feedback submitted")
        # lines I added to display your table
        query = pd.read_sql_query('''
                select * from feedback''', conn)

        data = pd.DataFrame(query)

        st.write(data)


# if __name__ == '__main__':
#     main()