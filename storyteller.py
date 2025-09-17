#!/usr/bin/env python

import os
import requests
import re
import jieba
import json
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Set, Dict, Tuple
from difflib import SequenceMatcher

API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"

class TitleAnalyzer:
    """智能标题分析器"""

    def __init__(self):
        # 常见的重复模式正则表达式
        self.repetitive_patterns = [
            r'^会.+的.+$',           # 会...的...
            r'^.+的.+声$',           # X的X声
            r'^.+的.+光$',           # X的X光
            r'^.+的.+夜$',           # X的X夜
            r'^.+的.+梦$',           # X的X梦
            r'^.+的.+歌$',           # X的X歌
            r'^.+里的.+$',           # X里的X
            r'^.+中的.+$',           # X中的X
            r'^最后的.+$',           # 最后的X
            r'^消失的.+$',           # 消失的X
            r'^神秘的.+$',           # 神秘的X
        ]

        # 过度使用的词汇
        self.overused_words = {
            '会', '的', '消失', '神秘', '最后', '古老', '奇怪', '魔法',
            '声音', '夜晚', '梦境', '记忆', '故事', '秘密', '时间'
        }

        # 标题质量权重
        self.quality_weights = {
            'length': 0.2,      # 长度适中
            'uniqueness': 0.3,  # 独特性
            'poetry': 0.2,      # 诗意性
            'specificity': 0.3  # 具体性
        }

    def extract_keywords(self, title: str) -> Set[str]:
        """提取标题关键词"""
        # 使用jieba分词
        words = jieba.lcut(title)
        # 过滤停用词和单字符
        keywords = {word for word in words if len(word) > 1 and word not in {'的', '了', '在', '是', '有'}}
        return keywords

    def calculate_similarity(self, title1: str, title2: str) -> float:
        """计算两个标题的相似度"""
        # 字符串相似度
        string_sim = SequenceMatcher(None, title1, title2).ratio()

        # 关键词相似度
        keywords1 = self.extract_keywords(title1)
        keywords2 = self.extract_keywords(title2)

        if not keywords1 or not keywords2:
            keyword_sim = 0
        else:
            intersection = len(keywords1 & keywords2)
            union = len(keywords1 | keywords2)
            keyword_sim = intersection / union if union > 0 else 0

        # 结构相似度（长度比较）
        len_sim = 1 - abs(len(title1) - len(title2)) / max(len(title1), len(title2))

        # 综合相似度
        return 0.4 * string_sim + 0.4 * keyword_sim + 0.2 * len_sim

    def is_repetitive_pattern(self, title: str) -> bool:
        """检查是否为重复模式"""
        for pattern in self.repetitive_patterns:
            if re.match(pattern, title):
                return True
        return False

    def calculate_title_quality(self, title: str, existing_titles: List[str]) -> float:
        """计算标题质量分数 (0-1)"""
        score = 0

        # 长度评分 (4-8个字符为最佳)
        length_score = 1 - abs(len(title) - 6) / 10
        length_score = max(0, min(1, length_score))

        # 独特性评分 (与现有标题的平均相似度)
        if existing_titles:
            similarities = [self.calculate_similarity(title, existing) for existing in existing_titles[-50:]]
            uniqueness_score = 1 - max(similarities) if similarities else 1
        else:
            uniqueness_score = 1

        # 诗意性评分 (避免过度使用的词汇)
        overused_count = sum(1 for word in self.overused_words if word in title)
        poetry_score = max(0, 1 - overused_count * 0.2)

        # 具体性评分 (避免过于抽象的词汇)
        abstract_words = {'东西', '事情', '地方', '时候', '样子', '感觉'}
        abstract_count = sum(1 for word in abstract_words if word in title)
        specificity_score = max(0, 1 - abstract_count * 0.3)

        # 综合评分
        score = (
            self.quality_weights['length'] * length_score +
            self.quality_weights['uniqueness'] * uniqueness_score +
            self.quality_weights['poetry'] * poetry_score +
            self.quality_weights['specificity'] * specificity_score
        )

        return score

    def is_title_acceptable(self, title: str, existing_titles: List[str], min_quality: float = 0.6) -> Tuple[bool, str]:
        """判断标题是否可接受"""
        # 检查重复模式
        if self.is_repetitive_pattern(title):
            return False, f"使用了重复模式: {title}"

        # 检查完全重复
        if title in existing_titles:
            return False, f"标题已存在: {title}"

        # 检查高相似度
        for existing in existing_titles[-30:]:  # 检查最近30个
            if self.calculate_similarity(title, existing) > 0.7:
                return False, f"与现有标题过于相似: {existing}"

        # 检查质量
        quality = self.calculate_title_quality(title, existing_titles)
        if quality < min_quality:
            return False, f"标题质量不达标: {quality:.2f} < {min_quality}"

        return True, f"质量评分: {quality:.2f}"


class SmartStoryGenerator:
    """智能故事生成器"""

    def __init__(self):
        self.analyzer = TitleAnalyzer()
        self.generation_history = []  # 记录生成历史

    def load_existing_titles(self) -> List[str]:
        """加载现有标题"""
        files = [x for x in os.listdir("_posts") if x.endswith(".md")]
        files.sort()
        return [x.split("-")[3].split(".")[0] for x in files]

    def analyze_title_trends(self, titles: List[str]) -> Dict:
        """分析标题趋势"""
        # 分析常见模式
        patterns = []
        for title in titles:
            for pattern in self.analyzer.repetitive_patterns:
                if re.match(pattern, title):
                    patterns.append(pattern)

        pattern_counts = Counter(patterns)

        # 分析常用词汇
        all_keywords = []
        for title in titles:
            all_keywords.extend(self.analyzer.extract_keywords(title))

        word_counts = Counter(all_keywords)

        return {
            'total_titles': len(titles),
            'common_patterns': dict(pattern_counts.most_common(5)),
            'overused_words': dict(word_counts.most_common(10)),
            'avg_length': sum(len(title) for title in titles) / len(titles) if titles else 0
        }

    def generate_enhanced_prompt(self, existing_titles: List[str], attempt: int = 1) -> str:
        """生成增强的提示词"""
        trends = self.analyze_title_trends(existing_titles)
        recent_titles = existing_titles[-20:] if len(existing_titles) > 20 else existing_titles

        # 根据尝试次数调整严格程度
        strictness_levels = [
            "请注意",
            "特别强调，严格",
            "最后警告，绝对"
        ]
        strictness = strictness_levels[min(attempt - 1, len(strictness_levels) - 1)]

        prompt = f"""请创作一个800字左右的魔幻现实主义短篇故事。

{strictness}要求：

1. 标题要求（极其重要）：
   - 绝对不要使用以下重复模式：{', '.join(trends['common_patterns'].keys())}
   - 避免过度使用词汇：{', '.join(list(trends['overused_words'].keys())[:5])}
   - 标题长度建议：4-8个汉字
   - 必须具体、独特、有诗意
   - 可参考风格：《午夜理发店》《石桥上的邮差》《阿婆的针线盒》

2. 严格禁止的标题类型：
   - "会...的..." 格式
   - "...的声音/光/梦/夜" 格式
   - "消失的..." "神秘的..." "最后的..." 格式
   - 过于抽象或常见的词汇

3. 不要使用这些已有标题：{', '.join(recent_titles[-10:])}

4. 故事内容：体现魔幻现实主义，融合超现实与日常生活

第{attempt}次尝试 - 请确保标题完全原创且富有特色。"""

        return prompt

    def extract_story_info(self, story_text: str) -> Tuple[str, str]:
        """提取故事标题和内容"""
        lines = [x.strip() for x in story_text.split("\n") if x.strip()]

        if not lines:
            raise ValueError("故事内容为空")

        # 提取标题（去除各种可能的标记）
        title_line = lines[0]
        title = re.sub(r'^[《「【\*]*|[》」】\*]*$', '', title_line).strip()

        # 提取内容（跳过空行）
        content_lines = []
        for line in lines[1:]:
            if line:  # 非空行
                content_lines.append(line)

        content = "\n".join(content_lines)

        if not title or not content:
            raise ValueError("标题或内容提取失败")

        return title, content

    def save_story(self, title: str, content: str) -> bool:
        """保存故事到文件"""
        try:
            filename = f"_posts/{datetime.now().strftime('%Y-%m-%d')}-{title}.md"

            with open(filename, "w", encoding='utf-8') as f:
                f.write(f"---\n")
                f.write(f"title: {title}\n")
                f.write(f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S +0800')}\n")
                f.write(f"---\n")
                f.write(f"\n")
                f.write(f"{content}\n")

            print(f"✅ 成功保存故事: {title}")
            return True

        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False

    def generate_story(self, max_attempts: int = 5) -> bool:
        """生成故事的主函数"""
        existing_titles = self.load_existing_titles()

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        for attempt in range(1, max_attempts + 1):
            print(f"\n🔄 第 {attempt} 次尝试生成故事...")

            # 生成提示词
            prompt = self.generate_enhanced_prompt(existing_titles, attempt)

            # API调用参数
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一位才华横溢的魔幻现实主义作家，擅长创作独特而富有诗意的作品。你从不重复使用相同的标题模式，每个故事都有独特的标题和深刻的内涵。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7 + (attempt - 1) * 0.1,  # 随尝试次数增加创造性
                "max_tokens": 2000,
                "top_p": 0.9
            }

            try:
                # 调用API
                response = requests.post(API_URL, json=data, headers=headers)
                result = response.json()

                if "choices" not in result:
                    print(f"❌ API调用失败: {result.get('error', '未知错误')}")
                    continue

                story_text = result["choices"][0]["message"]["content"]
                print(f"📝 生成的故事内容预览:\n{story_text[:200]}...")

                # 提取标题和内容
                title, content = self.extract_story_info(story_text)
                print(f"📑 提取的标题: {title}")

                # 检查标题质量
                is_acceptable, reason = self.analyzer.is_title_acceptable(title, existing_titles)

                if is_acceptable:
                    # 保存故事
                    if self.save_story(title, content):
                        print(f"🎉 成功！{reason}")
                        return True
                else:
                    print(f"❌ 标题不合格: {reason}")
                    if attempt < max_attempts:
                        print(f"🔄 准备第 {attempt + 1} 次尝试...")

            except Exception as e:
                print(f"❌ 处理过程出错: {e}")
                continue

        print(f"💥 经过 {max_attempts} 次尝试仍未成功生成合格的故事")
        return False


def main():
    """主函数"""
    if not API_KEY:
        print("❌ 错误: 请设置 DEEPSEEK_API_KEY 环境变量")
        return

    # 确保_posts目录存在
    os.makedirs("_posts", exist_ok=True)

    generator = SmartStoryGenerator()
    success = generator.generate_story(max_attempts=5)

    if not success:
        exit(1)


if __name__ == "__main__":
    main()