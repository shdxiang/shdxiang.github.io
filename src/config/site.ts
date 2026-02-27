export const siteConfig = {
  title: "xiaoxiang.io",
  tagline: "",
  description: "xiaoxiang.io",
  url: "https://xiaoxiang.io",
  lang: "zh-CN",
  author: {
    name: "Xiaoxiang"
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
