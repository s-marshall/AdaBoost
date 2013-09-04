require './boosted.rb'

class SevenSegmentDigitDisplayTest
  describe "Tests for DigitDisplay" do
  	before do
  		@display = DigitDisplay.new
  		@number_of_training_examples = 500
  	end
  	
    it "Creates training set" do
    	@display.createTrainingSet(@number_of_training_examples)
    	@display.training_set.length.should == @number_of_training_examples
    	@display.training_set[rand(@number_of_training_examples)][1].kind_of?(Fixnum).should == true
    	@display.training_set[rand(@number_of_training_examples)][0].length.should == 7
    end
    
    it "Gets the x values in the training set" do
    	@display.createTrainingSet(10)
    	x_values = @display.getX
    	x_values.length.should == 10
    	x_values[rand(10)].length.should == 7
    end

    it "Gets the y values in the training set" do
    	@display.createTrainingSet(10)
    	y_values = @display.getY
    	y_values.length.should == 10
    end
    
    it "Initializes the weights" do
    	@display.createTrainingSet(@number_of_training_examples)
    	@display.training_set.length.should == @number_of_training_examples
			@d_value = 1.0/(@number_of_training_examples * 9.0)
			@weight = Weights.new(Dimensions.new(10, @number_of_training_examples), @d_value, @display.getY)
			@weight.values.reduce(:+).reduce(:+).should be_within(0.001).of(1.0)
    end
    
    it "Gets random indices for matrix based on 2D pdf" do
			dims = Dimensions.new(3, 2)
			pdf_xy = [[0.15,0.15, 0.1],[0.1,0.2,0.3]]
			values_xy = [["a", "b", "x"], ["c", "d", "y"]]

			indices = get2DSampleIndices(dims, pdf_xy, 25)
			indices.length.should == 25
		
			samples = indices.map{|i| values_xy[i[1]][i[0]]}
			samples.map {|s| ["a", "b", "x", "c", "d", "y"].include? s}.reduce(:&).should == true
    end
  end
end

