import gc
import hashlib
import json
from pathlib import Path
from typing import List, Optional
import pandas as pd
from config import config
from src.graphrag.graph_enhancer import EnhancedGraphEnhancer
from src.graphrag.query_processor import QueryProcessor
from src.llm.interactor import LLMProcessor
from src.modules.AccuracyAnalysis import calculate_accuracies
from src.modules.MedicalQuestion import MedicalQuestion, SubGraph
from src.modules.filter import compare_enhanced_with_baseline
from src.modules.AccuracyAnalysis import intersect

def process_head_qa(json_path: str, output_path: str) -> None:
    """处理 HeadQA 数据集并保存到指定路径

    Args:
        json_path: HeadQA json文件路径
        output_path: 输出目录路径
    """
    logger = config.get_logger("head_processor")

    # 创建输出目录
    cache_dir = config.paths["cache"] / output_path / 'data' / 'original'
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 读取 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 读取所有的 .gold 文件到字典中
        gold_answers = {}
        gold_dir = Path(json_path).parent
        print(gold_dir)
        for gold_file in gold_dir.glob('*.gold'):
            exam_name = gold_file.stem  # 获取文件名（不含扩展名）
            gold_answers[exam_name] = {}

            with open(gold_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        qid, ans = line.strip().split()
                        gold_answers[exam_name][qid] = ans
                    except ValueError:
                        continue

        # 定义我们想要保留的主题
        valid_topics = {
            'B': 'Biology',
            'M': 'Medicine',
            'F': 'Pharmacy'
        }

        processed_count = 0

        # 遍历所有考试
        for exam_name, exam_data in data['exams'].items():
            # 获取主题代码
            topic_code = exam_name.split('_')[-1]
            if topic_code not in valid_topics:
                continue

            # 获取这个考试的答案
            exam_answers = gold_answers.get(exam_name, {})

            for q_data in exam_data['data']:
                try:
                    qid = q_data['qid']
                    # 检查是否有对应的gold答案
                    gold_answer = exam_answers.get(qid)
                    if gold_answer is None:
                        continue

                    # 格式化选项
                    options = {
                        f'op{chr(96 + answer["aid"])}': answer["atext"]
                        for answer in q_data['answers']
                    }

                    # 使用gold文件中的答案
                    correct_answer = f'op{chr(96 + int(gold_answer))}'

                    # 创建问题对象
                    question = {
                        "question": q_data['qtext'],
                        "is_multi_choice": True,
                        "correct_answer": correct_answer,
                        "options": options,
                        "topic_name": valid_topics[topic_code],
                        "context": None,
                        "initial_causal_graph": {
                            "nodes": [],
                            "relationships": [],
                            "paths": []
                        },
                        "causal_graph": {
                            "nodes": [],
                            "relationships": [],
                            "paths": []
                        },
                        "knowledge_graph": {
                            "nodes": [],
                            "relationships": [],
                            "paths": []
                        },
                        "enhanced_graph": {
                            "nodes": [],
                            "relationships": [],
                            "paths": []
                        },
                        "reasoning_chain": [],
                        "enhanced_information": "",
                        "analysis": "",
                        "answer": "",
                        "confidence": 0.0,
                        "chain_coverage": {
                            "success_counts": [],
                            "coverage_rates": [],
                            "total_successes": 0
                        },
                        "normal_results": []
                    }

                    # 使用问题文本的哈希作为文件名
                    filename = hashlib.md5(q_data['qtext'].encode()).hexdigest() + '.json'

                    # 保存到文件
                    with open(cache_dir / filename, 'w', encoding='utf-8') as f:
                        json.dump(question, f, indent=2, ensure_ascii=False)

                    processed_count += 1

                except Exception as e:
                    logger.error(f"Error processing question: {str(e)}")
                    continue

        logger.info(f"Successfully processed {processed_count} questions")

    except Exception as e:
        logger.error(f"Error processing HeadQA dataset: {str(e)}")

if __name__ == "__main__":
    head_qa_path = "../../cache/head-qa-es-en-pdfs/HEAD_EN/HEAD_EN.json"
    output_path = "head-qa-test"
    process_head_qa(head_qa_path, output_path)