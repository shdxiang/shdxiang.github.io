export const siteConfig = {
  title: "xiaoxiang.io",
  titleCased: "Xiaoxiang.io",
  tagline: "前沿论文收集与整理",
  description:
    "前沿论文收集与整理：机器学习、大模型、量子计算、机器人与物理学等方向的 arXiv 论文笔记与解读。",
  keywords: [
    "arXiv 论文笔记",
    "机器学习论文解读",
    "大模型",
    "量子计算",
    "机器人",
    "物理学",
    "论文精读",
    "research notes"
  ],
  url: "https://xiaoxiang.io",
  lang: "zh-CN",
  author: {
    name: "Xiaoxiang",
    url: "https://xiaoxiang.io"
  },
  social: {
    github: "https://github.com/shdxiang",
    twitter: "https://twitter.com/shdxiang",
    links: ["https://twitter.com/shdxiang", "https://github.com/shdxiang"]
  },
  analytics: {
    goatcounterId: "xiaoxiang"
  }
} as const;

export type SiteConfig = typeof siteConfig;
