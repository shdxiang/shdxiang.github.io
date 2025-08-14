#!/usr/bin/env python

import os
import requests

from datetime import datetime

API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"


def handle_story(story: str) -> None:
    lines = [x.strip() for x in story.split("\n")]

    title = lines[0][1:-1].strip("*").strip("《").strip("》")
    content = "\n".join(lines[2:]).strip()

    with open(f"_posts/{datetime.now().strftime('%Y-%m-%d')}-{title}.md", "w") as f:
        f.write(f"---\n")
        f.write(f"title: {title}\n")
        f.write(f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S +0800')}\n")
        f.write(f"---\n")
        f.write(f"\n")
        f.write(f"{content}\n")


def main() -> None:

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    files = [x for x in os.listdir("_posts") if x.endswith(".md")]
    files.sort()

    old_stories = [x.split("-")[3].split(".")[0] for x in files]
    old_stories = old_stories[-16:]

    user_prompt = f"请创作一个 1024 字左右的魔幻现实主义短篇故事, 只输出故事本身，不要输出任何其他内容。不要重复以下你已经讲过的故事: [{', '.join(old_stories)}]。"

    print("user_prompt:")
    print(user_prompt)

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位精通魔幻现实主义的作家"},
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": 0.7,
        "max_tokens": 1500,
    }

    response = requests.post(API_URL, json=data, headers=headers)
    result = response.json()

    if "choices" in result:
        story = result["choices"][0]["message"]["content"]
        print("story:")
        print(story)
        handle_story(story)
    else:
        print("failed to generate: ", result.get("error"))


if __name__ == "__main__":
    main()
