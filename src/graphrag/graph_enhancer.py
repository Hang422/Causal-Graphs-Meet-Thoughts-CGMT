from typing import List, Dict, Set, Tuple
import logging
from dataclasses import dataclass
import math

from src.modules.MedicalQuestion import MedicalQuestion
from src.graphrag.entity_processor import EntityProcessor


@dataclass
class PathScore:
    """Path scoring information"""
    path: str
    cui_match_score: float
    semantic_match_score: float
    length_score: float  # Normalized path length score (shorter is better)
    total_score: float
    path_length: int  # Original path length for reference


class EnhancedGraphEnhancer:
    def __init__(self, keep_ratio: float = 0.6):
        self.logger = logging.getLogger(__name__)
        self.entity_processor = EntityProcessor()
        self.keep_ratio = keep_ratio

    def enhance_graphs(self, question: MedicalQuestion) -> None:
        try:
            # Merge paths from both graphs
            all_paths = []
            all_paths.extend(question.causal_graph.paths)
            all_paths.extend(question.knowledge_graph.paths)

            # Group similar paths
            path_groups = {}
            for path in all_paths:
                entities, relations, intermediates = self._extract_path_elements(path)

                # Create key based on structure
                key = (
                    entities[0],  # start entity
                    entities[-1],  # end entity
                    tuple(sorted(set(intermediates)))  # unique sorted intermediate nodes
                )

                if key not in path_groups:
                    path_groups[key] = {
                        'paths': [],
                        'relations': [set() for _ in range(len(relations))]
                    }

                # Add path to group
                path_groups[key]['paths'].append(path)

                # Add relations at each position
                for i, rel in enumerate(relations):
                    path_groups[key]['relations'][i].add(rel)

            # Merge paths in each group
            merged_paths = []
            for key, group_info in path_groups.items():

                if len(group_info['paths']) > 1:
                    # Process group with multiple paths
                    merged = self._merge_path_group(group_info['paths'], group_info['relations'])
                    merged_paths.append(merged)
                else:
                    # Keep single path as is
                    merged_paths.append(group_info['paths'][0])


            # Score and select paths
            selected_paths = self._score_and_select_paths(merged_paths, question)
            question.enhanced_graph.paths = selected_paths


        except Exception as e:
            self.logger.error(f"Error in graph enhancement: {str(e)}", exc_info=True)
            question.enhanced_graph.paths = []

    def _extract_path_elements(self, path: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract entities, relations, and intermediates from a path"""
        parts = path.split('->')
        entities = []
        relations = []
        intermediates = []

        for i, part in enumerate(parts):
            # Extract entity
            if '(' in part and ')' in part:
                entity_part = part[part.find('(') + 1:part.find(')')]
                # Handle multiple entities
                curr_entities = [e.strip() for e in entity_part.split(' and ')]
                entities.extend(curr_entities)

                # Add to intermediates if not start/end
                if 0 < i < len(parts) - 1:
                    intermediates.extend(curr_entities)

            # Extract relation
            if '-' in part:
                relation = part.split('-')[1]
                relations.append(relation)

        return entities, relations, intermediates

    def _merge_path_group(self, paths: List[str], relations_by_pos: List[Set[str]]) -> str:
        """Merge a group of paths with similar structure"""
        parts_by_position = []
        example_path = paths[0]
        path_parts = example_path.split('->')

        # Initialize collection for each position
        for _ in range(len(path_parts)):
            parts_by_position.append({
                'entities': set(),
                'relations': set()
            })

        # Collect all entities and relations
        for path in paths:
            parts = path.split('->')
            for i, part in enumerate(parts):
                # Extract entities
                if '(' in part and ')' in part:
                    entity_part = part[part.find('(') + 1:part.find(')')]
                    for entity in entity_part.split(' and '):
                        parts_by_position[i]['entities'].add(entity.strip())

                # Extract relations
                if '-' in part:
                    relation = part.split('-')[1]
                    parts_by_position[i]['relations'].add(relation)

        # Build merged path
        merged_parts = []
        for i, part_info in enumerate(parts_by_position):
            if part_info['entities']:
                entities_str = ' and '.join(sorted(part_info['entities']))
                if part_info['relations']:
                    relations_str = '/'.join(sorted(part_info['relations']))
                    merged_parts.append(f"({entities_str})-{relations_str}")
                else:
                    merged_parts.append(f"({entities_str})")

        return '->'.join(merged_parts)

    def _score_and_select_paths(self, paths: List[str], question: MedicalQuestion) -> List[str]:
        """Score paths and select top ones based on relevance"""
        # Extract question entities
        question_cuis = self.entity_processor.extract_cuis_from_text(question.question)
        for option_text in question.options.values():
            question_cuis.update(self.entity_processor.extract_cuis_from_text(option_text))
        print(f"cuis:{question_cuis}")
        question_semantic_types = set()
        for cui in question_cuis:
            question_semantic_types.update(
                self.entity_processor.get_semantic_types_for_cui(cui))
        print(question_semantic_types)
        # Score paths
        path_scores = []
        for path in paths:
            path_cuis, path_stypes = self._extract_path_entities(path)
            path_length = len(path.split('->'))

            # Calculate scores
            cui_score = self._calculate_overlap_score(path_cuis, question_cuis)
            semantic_score = self._calculate_overlap_score(path_stypes, question_semantic_types)
            length_score = 1.0 / path_length  # Shorter paths get higher scores

            total_score = (cui_score * 0.4 + semantic_score * 0.4 + length_score * 0.2)
            path_scores.append((path, total_score))

        # Sort and select top paths
        path_scores.sort(key=lambda x: x[1], reverse=True)
        keep_count = max(1, int(len(paths) * self.keep_ratio))
        return [score[0] for score in path_scores[:keep_count]]

    def _extract_path_entities(self, path: str) -> Tuple[Set[str], Set[str]]:
        """Extract CUIs and semantic types from path"""
        entities = []
        parts = path.split('->')
        for part in parts:
            if '(' in part and ')' in part:
                entity = part[part.find('(') + 1:part.find(')')].strip()
                for sub_entity in entity.split(' and '):
                    entities.append(sub_entity.strip())

        path_cuis = set()
        for entity in entities:
            path_cuis.update(self.entity_processor.extract_cuis_from_text(entity))

        path_stypes = set()
        for cui in path_cuis:
            path_stypes.update(self.entity_processor.get_semantic_types_for_cui(cui))

        return path_cuis, path_stypes

    def _calculate_overlap_score(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate overlap score between two sets"""
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / len(set2)




def extract_path_elements(path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract entities, relations, and intermediate entities from a path
    Returns:
        Tuple of (entities, relations, intermediate_nodes)
    """
    parts = path.split('->')
    entities = []
    relations = []
    intermediate_nodes = []

    for i, part in enumerate(parts):
        # Extract main entity for this part
        if '(' in part and ')' in part:
            # Handle potential "and" separated entities
            entity_part = part[part.find('(') + 1:part.find(')')]
            main_entities = [e.strip() for e in entity_part.split(' and ')]

            # Add all entities at this position
            for entity in main_entities:
                entities.append(entity)
                if 0 < i < len(parts) - 1:  # If it's an intermediate node
                    intermediate_nodes.append(entity)

        # Extract relation if present
        if '-' in part:
            relation = part.split('-')[1]
            relations.append(relation)

    return entities, relations, intermediate_nodes


def merge_group(paths: List[str]) -> str:
    """
    Merge a group of paths with the same structure
    """
    if len(paths) == 1:
        return paths[0]

    parts_by_position = []
    example_path = paths[0]
    path_parts = example_path.split('->')

    # Initialize collection for each position
    for _ in range(len(path_parts)):
        parts_by_position.append({
            'entities': set(),
            'relation': None
        })

    # Collect all entities and relations at each position
    for path in paths:
        parts = path.split('->')
        for i, part in enumerate(parts):
            current = parts_by_position[i]

            # Extract entity
            if '(' in part and ')' in part:
                entity_part = part[part.find('(') + 1:part.find(')')]
                # Handle multiple entities separated by 'and'
                for entity in entity_part.split(' and '):
                    current['entities'].add(entity.strip())

            # Extract relation
            if '-' in part:
                relation = part.split('-')[1]
                if current['relation'] is None:
                    current['relation'] = relation
                elif current['relation'] != relation:
                    # If we find different relations, don't merge
                    return paths[0]

    # Build merged path
    merged_parts = []
    for i, part_info in enumerate(parts_by_position):
        if part_info['entities']:
            entities_str = ' and '.join(sorted(part_info['entities']))
            if part_info['relation']:
                merged_parts.append(f"({entities_str})-{part_info['relation']}")
            else:
                merged_parts.append(f"({entities_str})")

    return '->'.join(merged_parts)




def main1():
    """测试增强器功能"""
    # 创建测试用例
    question = MedicalQuestion(
        question= "Glycogen storage diseases include all the following except:",
    is_multi_choice= True,
    correct_answer= "opc",
    options= {
        "opa": "Amyloidosis associated with multiple myeloma has the poorest prognosis",
        "opb": "Fine - needle biopsy of subcutaneous abdominal fat is a simple & reliable method for diagnosing secondary systemic amyloidosis",
        "opc": "Hepatic amyloid disease produces hepatomegaly but rarely jaundice",
        "opd": "Amyloidosis of the spleen is associated with severe anemia"
        }
    )

    # 添加测试路径
    question.causal_graph.paths = [
        "(Amyloidosis)-CAUSES->(Disease)-CAUSES->(Splenomegaly)",
        "(Amyloidosis)-CAUSES->(Infection)-CAUSES->(Splenomegaly)",
        "(Amyloidosis)-CAUSES->(Hypertensive disease)-CAUSES->(Splenomegaly)",
        "(Splenomegaly)-CAUSES->(Anemia)",
        "(Spleen)-LOCATION_OF->(Splenomegaly)"
    ]

    question.reasoning_chain = [  "CHAIN: \"Amyloidosis associated with multiple myeloma\" -> \"generally poor prognosis\" -> \"widely accepted in medical literature\" -> 95%",
    "CHAIN: \"Fine-needle biopsy of subcutaneous abdominal fat\" -> \"used for diagnosing secondary systemic amyloidosis\" -> \"considered simple and reliable\" -> 90%",
    "CHAIN: \"Hepatic amyloid disease\" -> \"can cause hepatomegaly\" -> \"jaundice is uncommon\" -> 85%",
    "CHAIN: \"Amyloidosis of the spleen\" -> \"can lead to splenomegaly\" -> \"not typically associated with severe anemia\" -> \"conflict with known associations\" -> 70%"
    ]
    # 创建并运行增强器
    enhancer = EnhancedGraphEnhancer(keep_ratio=0.6)
    print("\nBefore enhancement:")
    print(f"Number of causal graph paths: {len(question.causal_graph.paths)}")
    print(f"Number of knowledge graph paths: {len(question.knowledge_graph.paths)}")

    enhancer.enhance_graphs(question)

    print("\nAfter enhancement:")
    print(f"Number of enhanced paths: {len(question.enhanced_graph.paths)}")
    print("\nEnhanced paths:")
    for path in question.enhanced_graph.paths:
        print(path)


def analyze_reasoning_chains(reasoning_chains: List[str], entity_processor: EntityProcessor) -> None:
    """分析思维链中的实体及其语义类型"""
    print("=== 思维链分析 ===\n")

    # 存储所有发现的CUI和语义类型
    all_cuis = set()
    all_semantic_types = set()

    for i, chain in enumerate(reasoning_chains, 1):
        print(f"\n分析推理链 {i}:")
        print(f"原始链: {chain}")

        # 移除"CHAIN:"前缀和置信度
        chain = chain.replace("CHAIN:", "").strip()
        steps = chain.split("->")
        steps = [step.strip() for step in steps[:-1]]  # 移除最后的置信度部分

        print("\n步骤分析:")
        for j, step in enumerate(steps, 1):
            # 清理步骤文本（移除引号等）
            step = step.strip('" ')
            print(f"\n步骤 {j}: {step}")

            # 使用extract_cuis_from_text提取CUIs
            step_cuis = entity_processor.extract_cuis_from_text(step)
            if step_cuis:
                print("发现的CUIs:")
                for cui in step_cuis:
                    all_cuis.add(cui)
                    print(f"  CUI: {cui}")

                    # 获取每个CUI的语义类型
                    semantic_types = entity_processor.get_semantic_types_for_cui(cui)
                    if semantic_types:
                        print(f"  语义类型: {semantic_types}")
                        all_semantic_types.update(semantic_types)
            else:
                print("  未发现CUIs")

            # 直接从文本中提取语义类型
            step_types = entity_processor.extract_semantic_types_from_text(step)
            if step_types:
                print(f"  直接从文本提取的语义类型: {step_types}")
                all_semantic_types.update(step_types)

    print("\n=== 总结 ===")
    print(f"\n发现的所有CUIs ({len(all_cuis)}):")
    for cui in sorted(all_cuis):
        print(f"CUI: {cui}")

    print(f"\n发现的所有语义类型 ({len(all_semantic_types)}):")
    for sem_type in sorted(all_semantic_types):
        print(f"语义类型: {sem_type}")


def analyze_paths(paths: List[str], entity_processor: EntityProcessor) -> None:
    """分析路径中的实体及其语义类型"""
    print("=== 路径分析 ===\n")

    # 存储所有唯一的实体及其信息
    entity_info = {}  # 用于保存每个实体名称的CUI和语义类型信息

    for i, path in enumerate(paths, 1):
        print(f"\n路径 {i}:")
        print(f"原始路径: {path}")

        # 提取和分析该路径中的所有实体
        entities = []
        parts = path.split('->')
        for part in parts:
            if '(' in part and ')' in part:
                entity_group = part[part.find('(') + 1:part.find(')')].strip()
                # 处理可能的多个实体
                for single_entity in entity_group.split(' and '):
                    entity = single_entity.strip()
                    entities.append(entity)

                    # 如果这个实体还没有被分析过
                    if entity not in entity_info:
                        cuis = entity_processor.extract_cuis_from_text(entity)
                        entity_info[entity] = {
                            'cuis': {},
                            'semantic_types': set()
                        }
                        # 对每个CUI获取语义类型
                        for cui in cuis:
                            semantic_types = entity_processor.get_semantic_types_for_cui(cui)
                            entity_info[entity]['cuis'][cui] = semantic_types
                            entity_info[entity]['semantic_types'].update(semantic_types)

        print("\n实体分析:")
        for entity in entities:
            print(f"\n实体: {entity}")
            if entity in entity_info and entity_info[entity]['cuis']:
                for cui, semantic_types in entity_info[entity]['cuis'].items():
                    print(f"  CUI: {cui}")
                    print(f"  语义类型: {semantic_types}")
            else:
                print("  未找到CUI和语义类型")

    # 打印总体统计
    print("\n=== 总体实体统计 ===")
    print(f"\n发现的所有独特实体 ({len(entity_info)}):")

    for entity, info in sorted(entity_info.items()):
        print(f"\n实体名称: {entity}")
        if info['cuis']:
            for cui, semantic_types in info['cuis'].items():
                print(f"  CUI: {cui}")
                print(f"  语义类型: {semantic_types}")
        else:
            print("  未找到CUI和语义类型")

    # 计算总的CUI和语义类型数量
    all_cuis = set()
    all_semantic_types = set()
    for info in entity_info.values():
        all_cuis.update(info['cuis'].keys())
        all_semantic_types.update(info['semantic_types'])

    print(f"\n总计:")
    print(f"独特实体总数: {len(entity_info)}")
    print(f"独特CUI总数: {len(all_cuis)}")
    print(f"独特语义类型总数: {len(all_semantic_types)}")

    print("\n所有发现的独特语义类型:")
    for sem_type in sorted(all_semantic_types):
        print(f"类型: {sem_type}")


# 使用示例
def main2():
    entity_processor = EntityProcessor(threshold=0.9)
    paths = [
      "(Hepatomegaly)-CAUSES->(Disease)-CAUSES->(Icterus)",
      "(Hepatomegaly)-CAUSES->(Respiratory distress)-CAUSES->(Icterus)",
      "(Hepatomegaly)-CAUSES->(Syndrome)-CAUSES->(Icterus)",
      "(Amyloidosis)-CAUSES->(Disease)-CAUSES->(Splenomegaly)",
      "(Amyloidosis)-CAUSES->(Infection)-CAUSES->(Splenomegaly)",
      "(Amyloidosis)-CAUSES->(Hypertensive disease)-CAUSES->(Splenomegaly)",
      "(Splenomegaly)-CAUSES->(Anemia)",
        "(Subcutaneous Fat, Abdominal)-PART_OF->(Subcutaneous Fat)-LOCATION_OF->(Diagnosis)",
        "(Fine-needle biopsy)-TREATS->(Symptoms)-CAUSES->(Sphincter)-LOCATION_OF->(Diagnosis)",
        "(Fine-needle biopsy)-TREATS->(Symptoms)-ASSOCIATED_WITH->(Sphincter)-LOCATION_OF->(Diagnosis)",
        "(Spleen)-LOCATION_OF->(Splenomegaly)"
    ]
    analyze_paths(paths, entity_processor)


if __name__ == "__main__":

    """语义类型: T044
语义类型: T047
语义类型: T169"""

    """类型: T046
类型: T047
类型: T079"""
    main2()
