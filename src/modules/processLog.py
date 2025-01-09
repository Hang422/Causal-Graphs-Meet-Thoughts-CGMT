import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from src.modules.AccuracyAnalysis import calculate_common_accuracies, calculate_accuracies
from src.modules.MedicalQuestion import MedicalQuestion
from config import config
from pathlib import Path
import shutil
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import logging
from dataclasses import asdict

@dataclass
class LogEntry:
    timestamp: str
    interaction_type: str
    response: dict
    metadata: dict
    question_id: str
    question_text: str


class DataRecovery:
    def __init__(self, logs_dir: Path, output_dir: Path):
        self.logs_dir = logs_dir
        self.output_dir = output_dir
        self.questions_by_stage = {}

        # 定义输出阶段映射
        self.stage_mapping = {
            'direct_answer': 'derelict',
            'answer_normal_rag': 'normal_rag',
            'answer_with_CoT': 'remove_llm_enhanced',
            'answer_with_enhancement_complete_with_chain': 'enhanced',
            'answer_with_enhancement_kg_only': 'knowledge_graph',
            'answer_with_enhancement_without_enhancer': 'remove_enhancer'
        }

        self.answer_stages = set(self.stage_mapping.keys())
        self.create_output_dirs()

    def create_output_dirs(self):
        """创建输出目录"""
        for stage in self.stage_mapping.values():
            (self.output_dir / stage).mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'original').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'reasoning').mkdir(parents=True, exist_ok=True)

    def parse_timestamp(self, timestamp: str) -> datetime:
        """解析时间戳"""
        try:
            return datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        except ValueError:
            return datetime.min

    def extract_reasoning_chains(self, response_text: str) -> List[str]:
        """提取推理链"""
        chains = []
        if isinstance(response_text, str):
            for line in response_text.split('\n'):
                if line.strip().startswith('CHAIN:'):
                    chains.append(line.strip())
        return chains

    def process_log_file(self, log_path: Path):
        """处理单个日志文件"""
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 处理响应
            response = data.get('response', '')
            if isinstance(response, str):
                try:
                    if response.strip().startswith('{'):
                        response = json.loads(response)
                    else:
                        response = {'raw_response': response}
                except json.JSONDecodeError:
                    response = {'raw_response': response}

            entry = LogEntry(
                timestamp=data.get('timestamp', ''),
                interaction_type=data.get('interaction_type', ''),
                response=response,
                metadata=data.get('metadata', {}),
                question_id=data.get('question_id', ''),
                question_text=data.get('question_text', '')
            )

            self.update_question_data(entry)

        except Exception as e:
            print(f"Error processing {log_path}: {str(e)}")

    def update_question_data(self, entry: LogEntry):
        """更新问题数据，只保留每个阶段最新的记录"""
        if not entry.interaction_type:
            return

        if entry.interaction_type not in self.questions_by_stage:
            self.questions_by_stage[entry.interaction_type] = {}

        stage_data = self.questions_by_stage[entry.interaction_type]
        current_timestamp = self.parse_timestamp(entry.timestamp)

        if entry.question_id not in stage_data or \
                current_timestamp > self.parse_timestamp(stage_data[entry.question_id][0]):
            stage_data[entry.question_id] = (entry.timestamp, entry)

    def process_log_directory(self, directory: Path):
        """递归处理目录下的所有日志文件"""
        if not directory.exists():
            return

        for item in directory.iterdir():
            if item.is_file() and item.suffix == '.json':
                self.process_log_file(item)
            elif item.is_dir():
                self.process_log_directory(item)

    def create_question(self, entry: LogEntry) -> MedicalQuestion:
        """从日志条目创建问题对象"""
        metadata = entry.metadata
        response = entry.response

        return MedicalQuestion(
            question=entry.question_text,
            is_multi_choice=metadata.get('is_multi_choice', True),
            options=metadata.get('options', {}),
            correct_answer=metadata.get('correct_answer', ''),
            topic_name=metadata.get('topic_name'),
            analysis=response.get('final_analysis', ''),
            answer=response.get('answer', ''),
            confidence=float(response.get('confidence', 0.0))
        )

    def save_question(self, question: MedicalQuestion, question_id: str, stage: str):
        """保存问题到指定目录"""
        save_path = self.output_dir / stage / f"{question_id}.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(question.to_dict(), f, ensure_ascii=False, indent=2)

    def reconstruct_data(self):
        """重建数据文件"""
        # 处理所有问题ID
        all_questions = set()
        for stage_data in self.questions_by_stage.values():
            all_questions.update(stage_data.keys())

        for question_id in all_questions:
            # 获取原始问题信息（从direct_answer）
            if 'direct_answer' in self.questions_by_stage and \
                    question_id in self.questions_by_stage['direct_answer']:
                _, entry = self.questions_by_stage['direct_answer'][question_id]
                question = self.create_question(entry)

                # 保存原始版本
                self.save_question(question, question_id, 'original')
                self.save_question(question, question_id, 'derelict')

                # 处理推理链
                if 'reasoning_chain' in self.questions_by_stage and \
                        question_id in self.questions_by_stage['reasoning_chain']:
                    _, reasoning_entry = self.questions_by_stage['reasoning_chain'][question_id]
                    question.reasoning_chain = self.extract_reasoning_chains(
                        reasoning_entry.response.get('raw_response', ''))
                    self.save_question(question, question_id, 'reasoning')

                # 处理每个答案阶段
                for stage, output_stage in self.stage_mapping.items():
                    if stage in self.questions_by_stage and \
                            question_id in self.questions_by_stage[stage]:
                        _, answer_entry = self.questions_by_stage[stage][question_id]
                        response = answer_entry.response

                        question.analysis = response.get('final_analysis', '')
                        question.answer = response.get('answer', '')
                        question.confidence = float(response.get('confidence', 0.0))

                        self.save_question(question, question_id, output_stage)



class QuestionCompleter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stages = [
            'original', 'derelict', 'reasoning', 'normal_rag',
            'remove_llm_enhanced', 'enhanced', 'knowledge_graph',
            'remove_enhancer'
        ]

    def load_questions_from_parquet(self, file_path: str, sample_size: Optional[int] = None) -> Dict[str, Dict]:
        """从parquet文件加载问题数据"""
        question_data = {}
        try:
            # 加载数据集
            splits = {
                'train': 'data/train-00000-of-00001.parquet',
                'validation': 'data/validation-00000-of-00001.parquet'
            }

            df = pd.read_parquet("hf://datasets/openlifescienceai/medmcqa/" + splits["train"])

            # 如果指定了样本大小，随机选择样本
            if sample_size is not None and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

            # 转换为字典格式
            for _, row in df.iterrows():
                question_data[row['question']] = {
                    'options': {
                        'opa': str(row['opa']),
                        'opb': str(row['opb']),
                        'opc': str(row['opc']),
                        'opd': str(row['opd'])
                    },
                    'correct_answer': (
                        'opa' if row['cop'] == 0 else
                        'opb' if row['cop'] == 1 else
                        'opc' if row['cop'] == 2 else
                        'opd' if row['cop'] == 3 else
                        None
                    ),
                    'topic_name': row['subject_name']
                }

        except Exception as e:
            self.logger.error(f"Error loading parquet file: {str(e)}")
            raise

        return question_data

    def update_question_file(self, file_path: Path, original_data: Dict[str, Dict]) -> None:
        """更新单个问题文件的选项和正确答案"""
        try:
            # 读取现有JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                question_data = json.load(f)

            # 获取问题文本
            question_text = question_data.get('question', '')
            if not question_text or question_text not in original_data:
                return

            # 更新数据
            original_info = original_data[question_text]
            question_data['options'] = original_info['options']
            question_data['correct_answer'] = original_info['correct_answer']
            question_data['topic_name'] = original_info['topic_name']

            # 保存更新后的文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(question_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.error(f"Error updating file {file_path}: {str(e)}")

    def update_project_questions(self, project_path: Path, original_data: Dict[str, Dict]) -> None:
        """更新项目目录下所有阶段的问题文件"""
        try:
            # 处理每个stage目录
            for stage in self.stages:
                stage_dir = project_path / stage
                if not stage_dir.exists():
                    continue

                self.logger.info(f"Processing stage: {stage}")
                for file_path in stage_dir.glob('*.json'):
                    self.update_question_file(file_path, original_data)

        except Exception as e:
            self.logger.error(f"Error processing project {project_path}: {str(e)}")


def insect_correct_answer():
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # 初始化
        completer = QuestionCompleter()

        # 加载原始数据
        logger.info("Loading original data from parquet file...")
        original_data = completer.load_questions_from_parquet("medmcqa")
        logger.info(f"Loaded {len(original_data)} questions from original dataset")

        # 定义项目列表
        projects = ["4-filtered", "4o-filtered", "4o-mini-filtered"]
        base_path = config.paths['cache']

        # 处理每个项目
        for project in projects:
            project_path = base_path / project / 'data'
            if not project_path.exists():
                logger.warning(f"Project directory not found: {project_path}")
                continue

            logger.info(f"Processing project: {project}")
            completer.update_project_questions(project_path, original_data)
            logger.info(f"Completed updating project: {project}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

def recover():
    base_logs_dir = Path("../../logs/llm_logs")
    projects = ['4-filtered','4o-filtered','4o-mini-filtered']

    for project in projects:
        print(f"\nProcessing project: {project}")
        logs_dir = base_logs_dir / project
        if not logs_dir.exists():
            print(f"Skipping non-existent project directory: {logs_dir}")
            continue

        output_dir = config.paths['cache'] / project / 'data'

        recovery = DataRecovery(logs_dir, output_dir)
        print(f"Processing logs from: {logs_dir}")
        print(f"Saving output to: {output_dir}")

        recovery.process_log_directory(logs_dir)
        recovery.reconstruct_data()
        print(f"Completed recovery for {project}")


def clean_json_response(response: str) -> dict:
    """清理和解析JSON响应"""
    if not response or not isinstance(response, str):
        return {"raw_response": str(response)}

    # 清理JSON代码块
    cleaned = response.strip()
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # 查找JSON对象
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}') + 1
    if start_idx == -1 or end_idx <= start_idx:
        return {"raw_response": response}

    json_str = cleaned[start_idx:end_idx]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 尝试清理和重新解析
        json_str = ' '.join(json_str.replace('\n', ' ').replace('\r', ' ').split())
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"raw_response": response}


def parse_response(data: dict) -> dict:
    """处理响应数据"""
    if 'response' not in data:
        return data

    response = data['response']
    if isinstance(response, dict):
        return data

    # 处理字符串形式的response
    data['response'] = clean_json_response(response)
    return data


def filter_logs(source_dir: Path, dest_dir: Path):
    """过滤日志文件，对每个阶段只保留每个问题ID的最新版本"""
    # 按阶段和问题ID分组存储文件
    files_by_stage_and_id = defaultdict(lambda: defaultdict(list))

    print(f"Processing directory: {source_dir}")

    # 遍历所有stage目录
    for stage_dir in source_dir.iterdir():
        if not stage_dir.is_dir():
            continue

        stage_name = stage_dir.name
        print(f"Processing stage: {stage_name}")

        # 遍历该stage下的所有json文件
        for json_file in stage_dir.rglob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 处理response字段
                data = parse_response(data)

                # 从文件名获取信息
                file_parts = json_file.stem.split('_')  # e.g., 20250103_010742_4a02a891
                if len(file_parts) != 3:
                    continue

                timestamp = f"{file_parts[0]}_{file_parts[1]}"
                question_id = file_parts[2]

                files_by_stage_and_id[stage_name][question_id].append({
                    'timestamp': timestamp,
                    'filepath': json_file,
                    'data': data
                })

            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue

    # 创建目标目录
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 处理每个阶段
    for stage, questions in files_by_stage_and_id.items():
        stage_dir = dest_dir / stage
        stage_dir.mkdir(exist_ok=True)

        # 对每个问题ID处理最新的文件
        for question_id, files in questions.items():
            if not files:
                continue

            # 按时间戳排序，获取最新的
            files.sort(key=lambda x: x['timestamp'], reverse=True)
            latest = files[0]

            # 保存处理后的数据
            dest_path = stage_dir / f"{latest['timestamp']}_{question_id}.json"
            with open(dest_path, 'w', encoding='utf-8') as f:
                json.dump(latest['data'], f, ensure_ascii=False, indent=2)


def filter():
    base_dir = Path("../../logs/llm_logs")
    projects = ['4', '4o', '4o-mini']

    for project in projects:
        source_dir = base_dir / project
        if not source_dir.exists():
            print(f"Skipping non-existent project: {project}")
            continue

        # 创建过滤后的目录
        dest_dir = base_dir / f"{project}-filtered"

        print(f"\nProcessing project: {project}")
        filter_logs(source_dir, dest_dir)
        print(f"Completed filtering for {project}")


if __name__ == "__main__":
    filter()
    recover()
    insect_correct_answer()
    projects = ['4-filtered', '4o-filtered', '4o-mini-filtered']
    for base_dir in projects:
        STAGES = ['derelict', 'enhanced', 'knowledge_graph', 'remove_llm_enhanced', 'normal_rag', 'remove_enhancer']
        df_report = calculate_common_accuracies(base_dir, STAGES)
        print(df_report)

        output_path = config.paths["output"] / f'{base_dir}.xlsx'
        df_report.to_excel(output_path, index=False)
