#!/usr/bin/env ruby
#
# Add :hash placeholder for MD5-based permalinks

require 'digest'

module Jekyll
  module Drops
    class UrlDrop
      def hash
        @obj.data['hash'] || ''
      end
    end
  end
end

Jekyll::Hooks.register :posts, :post_init do |post|
  # Calculate MD5 hash from post filename
  filename = File.basename(post.path)
  post.data['hash'] = Digest::MD5.hexdigest(filename)
end
