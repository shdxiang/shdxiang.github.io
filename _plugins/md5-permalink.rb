#!/usr/bin/env ruby
#
# Generate MD5-based permalinks for posts

require 'digest'

Jekyll::Hooks.register :posts, :pre_render do |post|
  # Calculate MD5 hash from post filename
  filename = File.basename(post.path)
  hash = Digest::MD5.hexdigest(filename)

  # Set hash
  post.data['hash'] = hash
end
