# Reddit_Data_Analysis

I've always wanted to examine reddit data, but I haven't been able to find a tool that allows you to examine specific discussions and find larger trends, without being super confusing and hard to use. So I rounded up some post/comment data from the r/Askdocs subreddit, (https://www.reddit.com/r/AskDocs/) and got to work!

My goal with this project was to create a program with a simple interface so that anyone could type in a term or phrase, and then Python would retrieve any posts/comments containing that phrase, and run some really simple sentiment analysis to tell the user how positive/negative it was on a scale of 10, using libraries like SentimentR (https://github.com/modarwish1/sentimentr). 

**Note**
I've been running my code locally due to the size of the data files, but I will commit my changes to this github going forward!

Development Progress:
 - Big milestone! My script now retrieves comments and gives each an (admittedly!) rough estimate on positivity/negativity.
 - 

Future Steps:
 -  My next steps will be moving my code to a platform like Google Collab, so that people could run my script (since the reddit files are pretty huge).
 -  Also I want to look into adding better sentiment models!
