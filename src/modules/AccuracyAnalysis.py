from pathlib import Path
from typing import Dict, List
import json
import pandas as pd
import logging
from config import config
import json
from pathlib import Path
from typing import Set, List, Dict, Tuple
import shutil
from collections import defaultdict
import hashlib
from config import config
logger = config.get_logger("question_processor")

def is_correct(q_data: Dict) -> bool:
    """Check if the answer matches correct_answer"""
    return q_data.get('answer', '').lower() == q_data.get('correct_answer', '').lower()


def load_questions_from_dir(dir_path: Path) -> Dict[str, Dict]:
    questions = {}
    if not dir_path.exists():
        logger.warning(f"Directory not found: {dir_path}")
        return questions

    for f in dir_path.glob("*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                q_data = json.load(fh)
            if 'question' in q_data and q_data['question']:
                questions[q_data['question']] = q_data
        except Exception as e:
            logger.error(f"Error loading {f}: {e}")
    return questions


def calculate_accuracies(base_dir: str, stages: List[str]) -> pd.DataFrame:

    base_path = config.paths["cache"] / base_dir / 'data'
    baseline_stage = "derelict"
    enhanced_stage = "enhanced"

    stage_questions = {}
    for stage in stages:
        stage_dir = base_path / stage
        stage_questions[stage] = load_questions_from_dir(stage_dir)

    baseline_qs = stage_questions.get(baseline_stage, {})
    enhanced_qs = stage_questions.get(enhanced_stage, {})

    # only for those questions with enhanced_graph (has query results)
    enhanced_filtered = {q: d for q, d in enhanced_qs.items() if d.get('enhanced_graph', {}).get('paths', []) and len( d.get('enhanced_graph', {}).get('paths', [])) > 0}
    filtered_questions = {q: baseline_qs[q] for q in enhanced_filtered if q in baseline_qs}

    if not filtered_questions:
        logger.warning("No questions found after filtering based on enhanced and intersecting with baseline.")
        return pd.DataFrame()
     
    baseline_total = len(filtered_questions)
    baseline_correct_set = {q for q, d in filtered_questions.items() if is_correct(d)}
    baseline_wrong_set = set(filtered_questions.keys()) - baseline_correct_set

    baseline_correct_num = len(baseline_correct_set)
    baseline_acc = (baseline_correct_num / baseline_total * 100) if baseline_total > 0 else 0.0

    results = []
    for stage in stages:
        q_map = stage_questions.get(stage, {})
         
        considered = {q: q_map[q] for q in filtered_questions.keys() if q in q_map}

        total_count = len(considered)
        if total_count == 0:
             
            results.append({
                'Model': stage,
                'Total_Questions': 0,
                'Overall_Accuracy(%)': 0.0,
                'Baseline_Correct_Count': len(baseline_correct_set),
                'Baseline_Correct_Accuracy(%)': 0.0,
                'Baseline_Wrong_Count': len(baseline_wrong_set),
                'Baseline_Wrong_Accuracy(%)': 0.0,
                'Improvement_over_Baseline(%)': 0.0
            })
            continue

        total_correct = sum(is_correct(d) for d in considered.values())
        overall_acc = (total_correct / total_count * 100)

        bc_questions = {q: considered[q] for q in baseline_correct_set if q in considered}
        bc_count = len(bc_questions)
        bc_correct = sum(is_correct(d) for d in bc_questions.values())
        bc_acc = (bc_correct / bc_count * 100) if bc_count > 0 else 0.0

        bw_questions = {q: considered[q] for q in baseline_wrong_set if q in considered}
        bw_count = len(bw_questions)
        bw_correct = sum(is_correct(d) for d in bw_questions.values())
        bw_acc = (bw_correct / bw_count * 100) if bw_count > 0 else 0.0

        improvement = overall_acc - baseline_acc

        results.append({
            'Model': stage,
            'Total_Questions': total_count,
            'Overall_Accuracy(%)': overall_acc,
            'Baseline_Correct_Count': bc_count,
            'Baseline_Correct_Accuracy(%)': bc_acc,
            'Baseline_Wrong_Count': bw_count,
            'Baseline_Wrong_Accuracy(%)': bw_acc,
            'Improvement_over_Baseline(%)': improvement
        })

    df = pd.DataFrame(results)
    return df


def load_questions_with_paths(enhanced_dir: Path) -> Dict[str, dict]:
    questions_with_paths = {}
    if not enhanced_dir.exists():
        print(f"Directory not found: {enhanced_dir}")
        return questions_with_paths

    for file in enhanced_dir.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if (data.get('enhanced_graph', {}).get('paths') and
                    len(data['enhanced_graph']['paths']) > 0):
                questions_with_paths[data['question']] = data
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return questions_with_paths


def find_intersection(model_dirs: List[Path]) -> Set[str]:
    """Find questions common to all directories"""
    questions_by_dir = {}
    for dir_path in model_dirs:
        enhanced_dir = dir_path / 'data' / 'enhanced'
        questions_with_paths = load_questions_with_paths(enhanced_dir)
        questions_by_dir[str(dir_path)] = set(questions_with_paths.keys())
        print(f"Found {len(questions_with_paths)} questions with paths in {dir_path}")

    common_questions = set.intersection(*questions_by_dir.values())
    print(f"\nFound {len(common_questions)} questions common to all directories")
    return common_questions


def copy_filtered_data(src_dir: Path, dest_dir: Path, common_questions: Set[str]) -> None:
    """Copy only intersection questions from source to destination directory"""
    if not src_dir.exists():
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    for file in src_dir.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data['question'] in common_questions:
                new_file = dest_dir / f"{hashlib.md5(data['question'].encode()).hexdigest()}.json"
                with open(new_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error processing file {file}: {e}")


def create_filtered_directories(base_dir: Path, model_dirs: List[str]) -> None:
    """Create filtered versions of directories containing only intersection questions"""
     
    model_paths = [base_dir / model_dir for model_dir in model_dirs]
    common_questions = find_intersection(model_paths)

     
    stages = ['original','derelict', 'enhanced', 'knowledge_graph', 'remove_llm_enhanced', 'normal_rag', 'remove_enhancer', 'reasoning']

    for model_dir in model_dirs:
        src_base = base_dir / model_dir
        dest_base = base_dir / f"{model_dir}-intersection"

        data_dir = dest_base / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)

        for stage in stages:
            src_stage = src_base / 'data' / stage
            dest_stage = data_dir / stage

            if src_stage.exists():
                print(f"Processing {model_dir}/{stage}")
                copy_filtered_data(src_stage, dest_stage, common_questions)


def intersect(model_dirs):
    base_dir = Path("/Users/luohang/PycharmProjects/casualGraphRag/cache")
    create_filtered_directories(base_dir, model_dirs)

    STAGES = ['derelict', 'enhanced', 'knowledge_graph', 'remove_llm_enhanced', 'normal_rag', 'remove_enhancer']

    model_dirs_new = [model_dir + '-intersection' for model_dir in model_dirs]
    for dir in model_dirs_new:
        base_dir = dir
        df_report = calculate_accuracies(base_dir, STAGES)
        print(df_report)

        output_path = config.paths["cache"] / base_dir / 'model_accuracy_report.xlsx'
        df_report.to_excel(output_path, index=False)


def load_directory_results(dir_path: Path) -> Dict[str, bool]:
    """Load results from all JSON files in a directory"""
    results = {}
    if not dir_path.exists():
        return results

    for json_file in dir_path.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                question_id = json_file.stem
                is_correct = data.get('answer', '').lower() == data.get('correct_answer', '').lower()
                results[question_id] = is_correct
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    return results


def find_common_questions(cache_path: Path) -> Set[str]:
    """Find intersection of questions across all directories"""
    # Load baseline questions
    baseline_questions = set(load_directory_results(cache_path / 'f1' / 'data' / 'derelict').keys())

    filtered_dirs = ['4-filtered', '4o-filtered', '4o-mini-filtered']
    subdirs = ['derelict', 'enhanced', 'knowledge_graph', 'remove_llm_enhanced',
               'normal_rag', 'remove_enhancer']

    all_questions = [baseline_questions]  # Start with baseline questions

    # Collect questions from each directory
    for filtered_dir in filtered_dirs:
        for subdir in subdirs:
            dir_path = cache_path / filtered_dir / 'data' / subdir
            results = load_directory_results(dir_path)
            if results:  # Only add if we got results
                all_questions.append(set(results.keys()))

    # Return intersection of all sets
    common_questions = set.intersection(*all_questions)
    return common_questions


def calculate_metrics(results: Dict[str, bool], baseline_results: Dict[str, bool],
                      common_questions: Set[str]) -> Dict:
    """Calculate metrics comparing against baseline"""
    tp = fp = fn = tn = correct = total = 0

    for qid in common_questions:
        if qid in results and qid in baseline_results:
            total += 1
            current = results[qid]
            baseline = baseline_results[qid]

            if current:
                correct += 1

            if current and baseline:
                tp += 1  # 都预测为正类，且正确
            elif current and not baseline:
                tn += 1  # current预测为负类但错误
            elif not current and baseline:
                fp += 1  # current预测为正类但错误
            else:  # not current and not baseline
                fn += 1  # 都预测为负类，且正确

    accuracy = (correct / total * 100) if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'Total_Questions': total,
        'Accuracy(%)': accuracy,
        'Recall(%)': recall * 100,
        'F1_Score': f1,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn
    }


def process_directories(cache_path: Path) -> pd.DataFrame:
    """Process specified directories and generate report"""
    # First find common questions across all directories
    common_questions = find_common_questions(cache_path)
    print(f"Found {len(common_questions)} questions common to all directories")

    # Load baseline results
    baseline_results = load_directory_results(cache_path / 'f1' / 'data' / 'derelict')

    # Hardcoded directories
    filtered_dirs = ['4-filtered', '4o-filtered', '4o-mini-filtered']
    subdirs = ['derelict', 'enhanced', 'knowledge_graph', 'remove_llm_enhanced',
               'normal_rag', 'remove_enhancer']

    # Calculate metrics for each directory
    all_results = []
    for filtered_dir in filtered_dirs:
        model_type = filtered_dir.replace('-filtered', '')
        data_dir = cache_path / filtered_dir / 'data'

        # Process each subdirectory
        for subdir in subdirs:
            subdir_path = data_dir / subdir
            subdir_results = load_directory_results(subdir_path)

            if subdir_results:  # Only process if we got results
                metrics = calculate_metrics(subdir_results, baseline_results, common_questions)

                result_row = {
                    'Model': model_type,
                    'Directory': subdir,
                    **metrics
                }
                all_results.append(result_row)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Sort by Model and Directory
    df = df.sort_values(['Model', 'Directory'])

    return df


def random_copy_json_files(src_dir: str, dest_dir: str, num_files: int) -> None:
    """
    Randomly select and copy a given number of JSON files from a source directory to a destination directory.

    :param src_dir: Path to the source directory containing JSON files.
    :param dest_dir: Path to the destination directory to store the files.
    :param num_files: Number of JSON files to copy.
    """
    # Ensure source and destination directories exist
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    if not src_path.is_dir():
        raise ValueError(f"Source directory does not exist: {src_dir}")

    # Get all JSON files in the source directory
    json_files = [file for file in src_path.iterdir() if file.suffix == '.json']
    target_files = [file for file in dest_path.iterdir() if file.suffix == '.json']
    dest_path = Path('../../cache/final')
    dest_path.mkdir(parents=True, exist_ok=True)
    # Randomly select the specified number of files
    j = 0
    for i in json_files:
        if i not in target_files:
            shutil.copy(i, dest_path)

            j += 1
        if j == num_files:
            break