# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jekyll-based personal blog using the Chirpy theme, hosted on GitHub Pages. The site features automated story generation using AI and includes bilingual content support.

## Core Architecture

- **Jekyll Static Site Generator**: Uses `jekyll-theme-chirpy` theme version ~7.3
- **Content Management**: Blog posts stored in `_posts/` directory with YAML front matter
- **Automated Content Generation**: Custom Python script for AI-powered story creation
- **GitHub Pages Deployment**: Static site deployment via GitHub Actions

## Key Components

### Content Structure
- `_posts/`: Blog posts in Markdown format with date-based naming (`YYYY-MM-DD-title.md`)
- `_tabs/`: Static pages (About, Archives, Categories, Tags)
- `_data/`: Site configuration data (contact info, social sharing)
- `_plugins/`: Jekyll plugins for post modification timestamps
- `assets/`: Static assets (images, favicons, CSS, JS)

### Automated Story Generation
- `storyteller.py`: Python script that generates magical realism stories using DeepSeek API
- `storyteller.sh`: Bash wrapper for automated git workflow (pull, generate, commit, push)
- Requires `DEEPSEEK_API_KEY` environment variable
- Tracks recent stories to avoid title duplication

### Configuration
- `_config.yml`: Main Jekyll configuration with site metadata, theme settings, and build options
- `Gemfile`: Ruby dependencies for Jekyll and theme
- Site URL: `https://xiaoxiang.io`
- Default language: English with timezone support

## Development Commands

### Local Development
```bash
# Install dependencies (requires Ruby/Bundler)
bundle install

# Serve locally with live reload
bundle exec jekyll serve

# Build for production
bundle exec jekyll build
```

### Content Generation
```bash
# Generate new story (requires DEEPSEEK_API_KEY)
python3 storyteller.py

# Full automated workflow
./storyteller.sh
```

### Git Workflow
```bash
# Standard workflow for content updates
git pull
git add _posts
git commit -am 'update posts'
git push
```

## Important Notes

- Posts use Chinese titles but are stored with date prefixes for Jekyll processing
- The story generator avoids duplicate titles by checking the last 64 posts
- Site uses PWA features and is optimized for mobile viewing
- GitHub Pages deployment is automatic on push to main branch
- No bundle command available in current environment - use alternative Jekyll setup if needed

## File Naming Conventions

- Blog posts: `YYYY-MM-DD-title.md` in `_posts/` directory
- Front matter includes title, date with timezone (+0800)
- Generated content follows magical realism theme with ~800 word stories