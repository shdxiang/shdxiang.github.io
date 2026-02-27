# xiaoxiang.io (Astro)

This site has been migrated from Jekyll/Chirpy to Astro.

## Stack

- Astro 5
- Markdown content from `_posts`
- `remark-gfm` + `remark-math`
- `rehype-katex`

## Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run check
npm run build
npm run preview
```

## Content model

- Existing post files stay in `_posts/*.md`
- Global site metadata lives in `src/config/site.ts`
- Frontmatter fields supported:
  - `title` (string)
  - `date` (date)
  - `arxiv_id` (optional string)
  - `tags` (optional string array)
  - `categories` (optional string array)
  - `description` (optional string)
  - `draft` (optional boolean)

## URL structure

- Home: `/`
- Posts: `/posts/YYYY/MM/DD/slug/`

`slug` is generated from the markdown filename after removing the `YYYY-MM-DD-` prefix.
