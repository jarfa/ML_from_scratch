# ML From Scratch

## A self-lead refresher in basic ML algorithms

This project was inspired by a hackathon this past Spring when I was working on the engineering team at Magnetic. At Magnetic I had been working on our machine learning pipelines that used [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit), and in this hackathon we attempted to implement a similar logistic regression solver in the Go language. My colleague Dan Crosta wrote about [what he learned about logistic regression](https://late.am/post/2016/04/22/demystifying-logistic-regression.html) from the process.

While I had learned a lot about the most commonly used algorithms in grad school, writing logistic regression from scratch and teaching a team of software engineers the math and intuition beyond the gradient descent solver made me think much harder about the various choices that go into writing a working implementation. It was a surprisingly educational experience.

 Since then I've changed jobs, and after a traveling a lot in the Summer and Fall I've found some free time again. I am (intermittently, on the occasional weekend) writing some of the more commonly used ML algorithms from near-scratch and comparing their performance (both in terms of predictive power and computational efficiency) versus [scikit-learn](http://scikit-learn.org/stable/). While various functions related to prediction metrics and data processing are also offered by scikit-learn, in many cases I'm also writing those from scratch when I think the implementation details are interesting.

 Note: these algorithms exist for my re-education and very little else. My algorithms will hopefully be just as good at prediction as scikit-learn's options, and theirs are more fully-featured and are much faster (since they're written in [Cython](http://cython.org/)). There is no reason anybody should be using my algorithms unless they find my code educational.
