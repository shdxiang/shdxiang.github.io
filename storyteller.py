#!/usr/bin/env python

import os
import requests
import re
from datetime import datetime
from collections import Counter

API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"


def extract_title_patterns(titles):
    """提取标题中的常见模式"""
    patterns = []
    for title in titles:
        # 检查"会...的..."模式
        if re.match(r'^会.+的.+', title):
            patterns.append("会...的...")
        # 检查其他可能的重复模式
        if '的' in title:
            parts = title.split('的')
            if len(parts) >= 2:
                patterns.append(f"{parts[0]}的...")
    return patterns


def check_title_similarity(new_title, existing_titles):
    """检查新标题是否与现有标题过于相似"""
    # 检查完全重复
    if new_title in existing_titles:
        return True

    # 检查"会...的..."模式
    if re.match(r'^会.+的.+', new_title):
        return True

    # 检查相似度过高的情况
    for existing in existing_titles[-20:]:  # 只检查最近20个
        if len(new_title) > 3 and len(existing) > 3:
            # 检查是否有相同的关键词组合
            new_words = set(re.findall(r'[\u4e00-\u9fff]+', new_title))
            existing_words = set(re.findall(r'[\u4e00-\u9fff]+', existing))
            if len(new_words & existing_words) >= 2:  # 有两个或以上相同词汇
                return True

    return False


def handle_story(story: str, existing_titles: list) -> bool:
    """处理故事，如果标题合适则保存文件，返回是否成功"""
    lines = [x.strip() for x in story.split("\n")]

    title = lines[0][1:-1].strip("*").strip("《").strip("》")
    content = "\n".join(lines[2:]).strip()

    # 检查标题是否合适
    if check_title_similarity(title, existing_titles):
        print(f"标题 '{title}' 与现有标题过于相似，跳过保存")
        return False

    with open(f"_posts/{datetime.now().strftime('%Y-%m-%d')}-{title}.md", "w") as f:
        f.write(f"---\n")
        f.write(f"title: {title}\n")
        f.write(f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S +0800')}\n")
        f.write(f"---\n")
        f.write(f"\n")
        f.write(f"{content}\n")

    print(f"成功保存故事: {title}")
    return True


def main() -> None:

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    files = [x for x in os.listdir("_posts") if x.endswith(".md")]
    files.sort()

    # 获取所有现有标题，不仅仅是最近64个
    all_titles = [x.split("-")[3].split(".")[0] for x in files]
    recent_titles = set(all_titles[-64:])  # 用于排除提示

    # 分析现有标题模式
    patterns = extract_title_patterns(all_titles)
    pattern_counts = Counter(patterns)
    common_patterns = [p for p, count in pattern_counts.items() if count > 3]

    # 构建更详细的提示词
    pattern_warning = ""
    if common_patterns:
        pattern_warning = f"特别注意：避免使用以下重复的标题模式：{', '.join(common_patterns)}。"

    user_prompt = f"""请创作一个 800 字左右的魔幻现实主义短篇故事。

要求：
1. 输出格式：标题和故事内容，不输出其他内容
2. 标题要求：
   - 避免使用"会...的..."句式
   - 要具有诗意和独特性
   - 可以使用人名、地点、物品、情感等作为标题
   - 标题示例风格：《雨中的理发店》《青石路的夜梦》《林默的蓝色指尖》《三楼右侧的守望者》
3. 不要使用以下已有标题：{', '.join(recent_titles)}
{pattern_warning}

故事应体现魔幻现实主义特色，将超现实元素与日常生活巧妙结合。"""

    print("user_prompt:")
    print(user_prompt)

    # 增加重试逻辑
    max_retries = 3
    for attempt in range(max_retries):
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一位精通魔幻现实主义的资深作家，擅长创作富有诗意和独创性的作品，从不重复使用相同的写作模式。"},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0.8,  # 稍微提高创造性
            "max_tokens": 2000,
        }

        response = requests.post(API_URL, json=data, headers=headers)
        result = response.json()

        if "choices" in result:
            story = result["choices"][0]["message"]["content"]
            print("story:")
            print(story)

            # 尝试保存故事
            if handle_story(story, all_titles):
                break  # 成功保存则退出重试循环
            elif attempt < max_retries - 1:
                print(f"第 {attempt + 1} 次生成的标题不合适，重新生成...")
                # 更新提示词，增加更严格的要求
                user_prompt += f"\n注意：刚才生成的标题不合适，请生成完全不同风格的标题。"
        else:
            print("failed to generate: ", result.get("error"))
            break


if __name__ == "__main__":
    main()
