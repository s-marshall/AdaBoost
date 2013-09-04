require "optparse"

class Dimensions
	attr_accessor :x, :y
	
	def initialize(x, y)
		@x = x
		@y = y
	end	
end 

class DigitDisplay 
	attr_reader :segments, :training_set, :test_set, :samples, :labels
	
	def initialize()
		@segments = [\
			[1,1,1,0,1,1,1],\
			[0,0,1,0,0,1,0],\
			[1,0,1,1,1,0,1],\
			[1,0,1,1,0,1,1],\
			[0,1,1,1,0,1,0],\
			[1,1,0,1,0,1,1],\
			[1,1,0,1,1,1,1],\
			[1,0,1,0,0,1,0],\
			[1,1,1,1,1,1,1],\
			[1,1,1,1,0,1,1]]
		
		@labels = [0,1,2,3,4,5,6,7,8,9]
		@training_set = []
		@test_set = []
		@samples = []
	end
	
	def createTrainingSet(count)
		createSet(count, 0.1, @training_set)
	end
	
	def createTestSet(count, error_rate)
		createSet(count, error_rate, @test_set)
	end
	
	def createSet(count, error_rate, set)
		probability_ON_is_ON = 1.0 - error_rate
		probability_OFF_is_ON = error_rate
		
		x = []
		y = []
		count.times do |i|
			y = rand(10) # label from 0 to 9
			correct_display = segments[y]
			x = correct_display.map do |n|
				if n == 1
					((rand < probability_ON_is_ON) && 1 ) || 0
				else
					((rand < probability_OFF_is_ON) && 1 ) || 0
				end
			end
			set << [x,y]
		end
	end
	
	def getX
		@training_set.map{|pair| pair.first}
	end

	def getY
		@training_set.map{|pair| pair.last}
	end
	
	def pickWeakLearnerSamples(indices)
		@samples = indices.map{|i| @training_set[i[1]]} 
	end
	
end

class Hypothesis
	attr_reader :number_of_elements_in_leaf0, :number_of_group_elements_in_leaf0, :number_of_elements_in_leaf1, :number_of_group_elements_in_leaf1
	
	def initialize
		@number_of_elements_in_leaf0 = []
		@number_of_group_elements_in_leaf0 = []

		@number_of_elements_in_leaf1 = []
		@number_of_group_elements_in_leaf1 = []
	end
                                                                     	
	def weakLearn(training_set)
		digit_segments = Array.new(7) {|i| i}
	
		digit_segments.each do |j|
			@@leaf0 = training_set.select{|digit, group| digit[j] == 0}
			@number_of_elements_in_leaf0[j] = @@leaf0.length.to_f
			@number_of_group_elements_in_leaf0[j] = (0..9).map{|group| @@leaf0.select{|set| set.last == group}.length.to_f}
	 
			@@leaf1 = training_set.select{|digit, group| digit[j] == 1} 
			@number_of_elements_in_leaf1[j] = @@leaf1.length.to_f
			@number_of_group_elements_in_leaf1[j] = (0..9).map{|group| @@leaf1.select{|set| set.last == group}.length.to_f}
		end
	end
	
	def compute(x, y)
		scores = []
		(0..9).each do |label|
			result = 0.0
			x.each_with_index do |segment_j, i|

				if segment_j == 0
					result += @number_of_group_elements_in_leaf0[i][label]/@number_of_elements_in_leaf0[i]
				end
		
				if segment_j == 1
					result += @number_of_group_elements_in_leaf1[i][label]/@number_of_elements_in_leaf1[i]
				end

			end
			scores << result
		end

		best_score = scores.max
		if scores.index(best_score) == y
			return 1
		else
			return 0
		end
	end
end

class Weights
	attr_reader :values
	
	@@no_rows = 0
	@@no_cols = 0
	
	def initialize(dimensions, value, ys)
		@values = Array.new(dimensions.y) { Array.new(dimensions.x) {|x| x = value}}
		@@no_rows = dimensions.y
		@@no_cols = dimensions.x
		@@no_rows.times do |i|
			@values[i][ys[i]] = 0.0
		end
	end
	
	def update(display, hypothesis, beta)
		updated_weights = Array.new(@@no_rows) {|x| Array.new(@@no_cols)}
	
		@@no_rows.times do |i|
			x = display.training_set[i][0]
			y = display.training_set[i][1]
		
			display.labels.each do |j|
				if y == j
					@values[i][j] = 0.0
				else
					@values[i][j] = @values[i][j] * beta**(0.5 * (1.0 + hypothesis.compute(x, y).to_f - hypothesis.compute(x, j).to_f))
				end
			end
		end
	
		@values = normalizeMatrix(@values)
	end

	private
	def normalizeMatrix(matrix)
 		normalization_constant = sumOverMatrix(matrix)
		matrix.map{|row| row.map{|element| element/normalization_constant}}
	end

	def sumOverMatrix(values)
		values.reduce(:+).reduce(:+)
	end
end

def calculatePseudoloss(weights, display, hypothesis)
	loss = 0.0
	
	display.training_set.each_with_index do |training_sample, i|
		x = training_sample[0]
		y = training_sample[1]
		
		correct_hypothesis = hypothesis.compute(x, y)
		display.labels.each do |label|
			loss += weights[i][label] * (1.0 - hypothesis.compute(x, y).to_f + hypothesis.compute(x, label).to_f)
		end
	end

	0.5 * loss
end

def calculateBeta(loss)
	loss/(1.0 - loss)
end

def get2DSampleIndices(dimensions, probability_distribution, no_samples)
	convertToTwoDIndex = ->(oneDIndex, dimensions) do
		y = oneDIndex/dimensions.x
		x = oneDIndex - y * dimensions.x 
		[x, y]
	end
	
	pdf = probability_distribution.flatten
	
	indices = []
	no_samples.times do |s|
		sample_probability = rand
		idx = 0
		cumulative_probability = pdf[idx]
		while (cumulative_probability < sample_probability)
			idx += 1
			cumulative_probability += pdf[idx]
		end
		indices << convertToTwoDIndex.call(idx, dimensions)
	end

	indices
end

def finalHypothesis(x, ys, hypotheses, betas)
	maximum_score = -1.0
	max_arg = -1
	
	number_of_rounds = hypotheses.length
	
	ys.each do |y|
		result = 0.0
		number_of_rounds.times do |round|
			result += Math.log(1.0/betas[round]) * hypotheses[round].compute(x, y)
		end
				
		if result >= maximum_score
			maximum_score = result
			max_arg = y
		end
	end
	
	max_arg
end

# ruby boosted.rb [-t number_of_training_examples] [-w number_of_weak_learners] [-b number_of_boosting_iterations]

number_of_training_examples = 2000
number_of_samples_for_weak_learner = 80
number_of_boosting_rounds = 200

OptionParser.new {|options|
	options.on("-t [int]", "# number of training examples"){|value|
		number_of_training_examples = value.to_i
	}

	options.on("-w [int]", "# number of weak learners"){|value|
		number_of_samples_for_weak_learner = value.to_i
	}

	options.on("-b [int]", "# number of boosting iterations"){|value|
		number_of_boosting_rounds = value.to_i
	}
	options.parse!(ARGV)
}

display = DigitDisplay.new
puts "Creating #{number_of_training_examples} training examples..."
display.createTrainingSet(number_of_training_examples)

d_value = 1.0/(number_of_training_examples * 9.0)
weight = Weights.new(Dimensions.new(10, number_of_training_examples), d_value, display.getY)

hypotheses = []
betas = []
dimensions = Dimensions.new(10, number_of_training_examples)

number_of_boosting_rounds.times do |round|
	puts "ROUND #{round}"
	
	indices = get2DSampleIndices(dimensions, weight.values, number_of_samples_for_weak_learner)

	display.pickWeakLearnerSamples(indices)

	hypotheses[round] = Hypothesis.new
	hypotheses[round].weakLearn(display.samples)

	pseudoloss = calculatePseudoloss(weight.values, display, hypotheses[round])
	
	puts "Pseudoloss #{pseudoloss}"
	puts
	if (pseudoloss >= 0.5)
		hypotheses.pop
		break
	end

	betas[round] = calculateBeta(pseudoloss)

	weight.update(display, hypotheses[round], betas[round])
end

puts
puts "Testing final hypothesis on perfect inputs..."
puts "input == or != guess"
10.times do |y|
	guess = finalHypothesis(display.segments[y], display.labels, hypotheses, betas)
	if y == guess
		puts "#{y} == #{guess}"
	else
		puts "#{y} != #{guess}"
	end
end

# Test
puts "Testing final hypothesis on inputs with errors..."
[0.01, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25].each do |test_error|
	puts "Creating test set with segment error rate of #{test_error}..."
	display.createTestSet(1000, test_error)

	puts "\tFinding error rate..."
	number_of_errors = 0.0
	total = 0.0
	display.test_set.each do |example|
	
		x = example.first
		y = example.last
		
	 	guess = finalHypothesis(x, display.labels, hypotheses, betas)
	 	if guess != y
	 		number_of_errors += 1.0
	 	end
	 	total += 1.0
	end

	puts "\tError rate: #{number_of_errors/total}\n"
end


