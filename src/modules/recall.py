import json
from pathlib import Path
from typing import Dict, Set
import pandas as pd


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


def main():
    cache_path = Path('../../cache')
    df = process_directories(cache_path)

    # Save detailed results
    output_path = cache_path / 'metrics_report_detailed.xlsx'
    df.to_excel(output_path, index=False)

    # Create simplified version for display
    display_df = df[['Model', 'Directory', 'Total_Questions', 'Accuracy(%)', 'Recall(%)', 'F1_Score']]

    # Save simplified results
    simple_output_path = cache_path / 'metrics_report.xlsx'
    display_df.to_excel(simple_output_path, index=False)

    print(f"\nDetailed results saved to {output_path}")
    print(f"Simplified results saved to {simple_output_path}")
    print("\nMetrics Summary:")
    print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()