# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Astro-based personal blog hosted on GitHub Pages. The site content is Markdown-first and posts are stored in `_posts/`.

## Core Architecture

- **Astro Static Site Generator**: Astro 5 static output
- **Content Management**: Blog posts stored in `_posts/` directory with front matter
- **Rendering Pipeline**: `remark-gfm`, `remark-math`, and `rehype-katex`
- **GitHub Pages Deployment**: Build and deploy via GitHub Actions

## Key Components

### Content Structure
- `_posts/`: Blog posts in Markdown format with date-based naming (`YYYY-MM-DD-title.md`)
- `assets/`: Static assets (images, favicons, CSS, JS)
- `public/`: Static passthrough files (`CNAME`, `.nojekyll`)
- `src/`: Astro source code (layouts, pages, styles, config)

### Configuration
- `src/config/site.ts`: Site metadata (title, description, social, analytics)
- `astro.config.mjs`: Astro config and markdown plugin setup
- Site URL: `https://xiaoxiang.io`
- Default language: `zh-CN`

## Development Commands

### Local Development
```bash
# Install dependencies
npm install

# Start local dev server
npm run dev

# Build for production
npm run build
```

### Git Workflow
```bash
# Standard workflow for content updates
git pull
git add _posts
git commit -m 'update posts'
git push
```

## Important Notes

- Posts use date-prefixed filenames and are mapped to `/posts/<md5(filename)>/`
- Existing `_posts` content is loaded directly by Astro content collections
- Site includes KaTeX support for math rendering in markdown
- GitHub Pages deployment is automatic on push to main branch

## File Naming Conventions

- Blog posts: `YYYY-MM-DD-title.md` in `_posts/` directory
- Front matter includes title, date with timezone (+0800)
