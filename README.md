# N-gram-text-analysis
Explored the use of n-grams to analyze textual similarity with SpaCy

To further investigate the Austen texts, I developed two functions that can help us better visualize the distribution of word frequency in the three Austen texts provided. With the first function, I generated one distribution plot for each Austen text to investigate what the most common token length is: 


Above is the distribution plot of token length and its frequency in the Emma text. 
We get from the graph that the max frequency of the token is almost 0.7 in the text, indicating the token with a length around 1 appeared almost 70% of the time in the text. As we can also see, there is a huge difference in frequency between short tokens of length before and after 5. This makes sense because the usual length of a word in English is 4.7 characters. It is also not surprising to see there are almost 0 tokens of length greater than 15 because those lengths are usually too long to appear as a common expression in novels. 
Additionally, we can observe the trend in which the frequency decreases as the token length increases, corresponding to Zipf’s law. 


The second graph is generated from the data of Sense. As we can see, it follows a similar pattern as the first figure but its most frequent token has a slightly lower frequency compared to the data in Emma. It is interesting to notice that the frequency does not strictly follow a decreasing pattern as the frequency of the token length around 3 is higher than that of the token with a shorter length. We may guess that such tokens would be frequent words like “the”.




The third figure below shows the distribution plot of tokens in the Persuasion text. As we can tell, the most common token length still lies in the range between 2.5 to 5. The plot always decreases to 0 when the token length is more than 15 in all three graphs, which validates that a word of length more than 15 is very unusual. 


Additionally, even if we are just comparing the length and not the exact word, we can have a sense that the writing style may not be very strictly consistent as the three graphs are of shape a bit different than each other. 

To have a better sense of how the token length relates to its frequency in regard to all texts in guten-berg, I created another function to plot a graph that allows comparisons over all texts. 



As the unit on the y-axis indicates, we have way more appearances of tokens with all texts combined together. The difference between each level of token length is still very significant. The graph generally still follows the pattern where token length is inversely related to its frequency. However, it is interesting to note that when the token length grows big enough, it appears with a frequency lower than 0.000001. 
