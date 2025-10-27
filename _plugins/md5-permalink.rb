#!/usr/bin/env ruby
#
# Generate MD5-based permalinks for posts

require 'digest'

Jekyll::Hooks.register :posts, :pre_render do |post|
  # Calculate MD5 hash from post filename
  filename = File.basename(post.path)
  md5_hash = Digest::MD5.hexdigest(filename)

  # Set the permalink directly
  post.data['permalink'] = "/posts/#{md5_hash}/"
end
