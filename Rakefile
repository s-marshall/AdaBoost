require 'rake'
require 'rspec/core/rake_task'

RSpec::Core::RakeTask.new do |c|
  options = ['--color']
  options += ["--format", "documentation"]
  c.rspec_opts = options
end

task :default => :spec
