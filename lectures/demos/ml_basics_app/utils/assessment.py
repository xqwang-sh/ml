import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import time
from datetime import datetime

class LearningAssessment:
    """学习进度评估系统"""
    
    def __init__(self, user_id="default_user"):
        self.user_id = user_id
        self.progress_dir = os.path.join("data", "user_progress")
        os.makedirs(self.progress_dir, exist_ok=True)
        self.progress_file = os.path.join(self.progress_dir, f"{user_id}_progress.json")
        self._load_progress()
    
    def _load_progress(self):
        """加载用户学习进度"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                self.progress_data = json.load(f)
        else:
            # 初始化进度数据
            self.progress_data = {
                "theory": {
                    "logistic_regression": 0,
                    "svm": 0
                },
                "exercises": {
                    "basic_1": 0,
                    "basic_2": 0,
                    "advanced_1": 0
                },
                "quizzes": {},
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self._save_progress()
    
    def _save_progress(self):
        """保存用户学习进度"""
        self.progress_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress_data, f, ensure_ascii=False, indent=2)
    
    def update_theory_progress(self, section, value):
        """更新理论学习进度"""
        if section in self.progress_data["theory"]:
            self.progress_data["theory"][section] = value
            self._save_progress()
    
    def update_exercise_progress(self, exercise_id, value):
        """更新练习完成进度"""
        if exercise_id in self.progress_data["exercises"]:
            self.progress_data["exercises"][exercise_id] = value
            self._save_progress()
    
    def record_quiz_result(self, quiz_id, score, total):
        """记录测验结果"""
        self.progress_data["quizzes"][quiz_id] = {
            "score": score,
            "total": total,
            "percentage": round(score / total * 100, 1),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_progress()
    
    def get_overall_progress(self):
        """获取总体学习进度百分比"""
        theory_progress = sum(self.progress_data["theory"].values()) / len(self.progress_data["theory"])
        exercise_progress = sum(self.progress_data["exercises"].values()) / len(self.progress_data["exercises"])
        
        quiz_progress = 0
        if self.progress_data["quizzes"]:
            quiz_scores = [data["percentage"] for data in self.progress_data["quizzes"].values()]
            quiz_progress = sum(quiz_scores) / len(quiz_scores) / 100
        else:
            quiz_progress = 0
        
        # 加权计算总进度
        overall = 0.3 * theory_progress + 0.4 * exercise_progress + 0.3 * quiz_progress
        return round(overall * 100, 1)
    
    def show_progress_dashboard(self):
        """显示学习进度仪表板"""
        st.subheader("学习进度")
        
        overall_progress = self.get_overall_progress()
        
        # 进度条
        st.progress(overall_progress / 100)
        st.write(f"总体完成度: {overall_progress}%")
        
        # 详细进度
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 理论学习")
            for section, value in self.progress_data["theory"].items():
                section_name = "逻辑回归" if section == "logistic_regression" else "支持向量机"
                st.write(f"{section_name}: {int(value * 100)}%")
            
            st.markdown("##### 练习完成")
            for ex_id, value in self.progress_data["exercises"].items():
                if ex_id == "basic_1":
                    ex_name = "基础练习1: 乳腺癌数据分类"
                elif ex_id == "basic_2":
                    ex_name = "基础练习2: 鸢尾花分类"
                else:
                    ex_name = "综合练习: 收入预测"
                
                st.write(f"{ex_name}: {int(value * 100)}%")
        
        with col2:
            st.markdown("##### 测验成绩")
            if self.progress_data["quizzes"]:
                for quiz_id, data in self.progress_data["quizzes"].items():
                    st.write(f"{quiz_id}: {data['score']}/{data['total']} ({data['percentage']}%)")
            else:
                st.write("尚未完成任何测验")
        
        st.info(f"最后更新时间: {self.progress_data['last_updated']}")

def create_quiz(quiz_id, questions, show_result=True):
    """创建一个测验并处理用户回答
    
    参数:
    - quiz_id: 测验ID
    - questions: 问题列表，每个问题是一个字典，包含question, options和correct_index
    - show_result: 是否立即显示结果
    
    返回:
    - score: 用户得分
    - total: 总分
    """
    st.subheader(f"测验: {quiz_id}")
    
    total = len(questions)
    
    # 初始化会话状态
    if f"answers_{quiz_id}" not in st.session_state:
        st.session_state[f"answers_{quiz_id}"] = [questions[i]['options'][0] for i in range(total)]
    
    if f"submitted_{quiz_id}" not in st.session_state:
        st.session_state[f"submitted_{quiz_id}"] = False
        
    if f"score_{quiz_id}" not in st.session_state:
        st.session_state[f"score_{quiz_id}"] = 0
    
    # 定义提交回调函数
    def submit_quiz():
        st.session_state[f"submitted_{quiz_id}"] = True
        
        # 计算得分
        score = 0
        for i, ans in enumerate(st.session_state[f"answers_{quiz_id}"]):
            if ans == questions[i]['options'][questions[i]['correct_index']]:
                score += 1
        
        st.session_state[f"score_{quiz_id}"] = score
        
        # 记录测验结果
        assessment = LearningAssessment()
        assessment.record_quiz_result(quiz_id, score, total)
    
    # 实现问题
    for i, q in enumerate(questions):
        st.markdown(f"**问题 {i+1}**: {q['question']}")
        
        # 添加"请选择"作为第一个虚拟选项
        answer_key = f"q_{quiz_id}_{i}"
        
        # 显示下拉选择框，确保总是有一个合法值
        selected = st.selectbox(
            label=f"选择答案 (问题 {i+1})",
            options=q['options'],
            index=q['options'].index(st.session_state[f"answers_{quiz_id}"][i]) if st.session_state[f"answers_{quiz_id}"][i] in q['options'] else 0,
            key=answer_key
        )
        
        # 保存选择
        st.session_state[f"answers_{quiz_id}"][i] = selected
    
    # 提交按钮
    st.button("提交答案", on_click=submit_quiz, key=f"submit_{quiz_id}")
    
    # 显示结果
    if st.session_state.get(f"submitted_{quiz_id}", False):
        score = st.session_state[f"score_{quiz_id}"]
        
        if show_result:
            st.success(f"你的得分: {score}/{total} ({round(score/total*100, 1)}%)")
            
            # 显示正确答案
            st.markdown("#### 答案解析")
            for i, ans in enumerate(st.session_state[f"answers_{quiz_id}"]):
                q = questions[i]
                correct_answer = q['options'][q['correct_index']]
                
                if ans == correct_answer:
                    st.markdown(f"✅ **问题 {i+1}**: 回答正确")
                else:
                    st.markdown(f"❌ **问题 {i+1}**: 你的回答: {ans}。正确答案是: {correct_answer}")
                
                # 如果有解释，显示解释
                if "explanation" in q:
                    st.markdown(f"解释: {q['explanation']}")
                st.markdown("---")
        
        return score, total
    
    return None, total 