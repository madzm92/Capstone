# Capstone
Capstone Project

In addition to general documentation,

Here I will provide weekly status updates on my project

## Project Mission Statement

The goal of this project is to utilize a variety of data science, and software engineering techniques to build a website that brings some benefit to a group of people (to be decided after project idea submission). In this project I would like to use: 
- Artificial Intelligence: This could be designing a machine learning model, using NLP, or building a recommendation algorithm
- Design a website a user can interact with: Allow a user to input something into a website, and receive something in return

## Week 1: Jan/21 - Jan/27
- Introduction: Discuss general details and guidlines of project
- Deliverable: Set up status tracker & Project Ideas

### Failures:
 
 No failures yet.

### Successes:

Enjoyed brainstorming project ideas, and am excited to get started.

### Difficulties: 

Had a hard time designing a project that will incorperate everything I want to accomplish.

### Goals for Next Week

My main goal for next week is to choose a project idea, and work on the design doc. For the design doc I would like to:
- Create a general outline of what I would like to include in the project, and by which dates I will need to complete everything

Due: Feb/4
- Create the requirement specification slides: include high level requirements, target audience & users, and an engineering diagram of inputs and outputs of project 


Project Ideas

Idea 1: 

Build a website that compares a job description to a resume, and provides a score as to how well matched the resume is to the job description:
- Utilize NLP to identify most important words in the job description, and check if the resume has these
- Build an algorithm to identify how well matched the resume is to the job description
- Additional Feature: Use webscraping or find an API to ingest job description data, and extract most frequently used skills associated with job titles. Allow the user to filter for job titles and view top skills associated with each title

Idea 2:

Product Price Tracker: Vehicles
- Ingest data for current cars available for purchase, and recommend the best vehicle to purchase. Build the recommender algorithim using price, miles, months on lot etc. Identify specific factors that would bring the value of the vehicle down, and provide these to the user to help negotiate a lower price. 

Idea 3: 

Predict which Tech Skills will be most relevant in the future
- Utilize historical job description data for tech jobs, and predict which skills & technologies will become more important in the next five year

## Week 2: Jan/28 - Feb/3
- Deliverable: None 

### Failures:
Continued research on idea one. My main goal is to get idea one working, but I need to find some data sources that work to validate the model. I have found one article that references two different data sets, but I am having issues accessing them. I will spend the next day or so doing some addional research, and if I cannot find a data source to use, I will pivot to a new idea. 

### Successes: 

Finding the one article that references sources.

### Difficulties: 

Coming up with alternative ideas that would involve easier data sources.

## Week 3: Feb/4 - Feb/10
- Deliverable: Create the requirement specification slides: include high level requirements, target audience & users, and an engineering diagram of inputs and outputs of project 

### Successes: 
Successfully chose a topic to work on! I decided to do my project on the MBTA Communities Law, and explore the effects of this law on certian topics such as traffic, infrustructure, schools, and property value. 

### Failures:
No failures this week

### Difficulties: 
MassDot data I found is very granular, but unfortunately does not include many data points. There are many locations that provide data, but from the 2 I have looked at so far, there were only 2 days worth of data for the entire year for 1 or 2 years. I still think that this data is usable because of the extensive amounts of traffic locations, but it will involve work to make the data usable. 

For next week, I plan to finalize the data sources I would like to use for this project, and begin to ingest & store the data or create a plan how I would like to ingest the data. Some data is easy to access in the form of a csv file. Some data, like the MassDot data does not seem to have export capabilities. I might try and reach out to the website to see if they have an api. Alternitively it would be a rather manual process to retreive the data, but not impossible. 

### Data Sources Identified So Far:
- Law Info: https://www.mass.gov/info-details/mbta-communities-law-qa#:~:text=The%20MBTA%20Communities%20Act%20requires,%2C%20ferry%20terminal%20or%20subway
- MassDot: Provides total number of cars that pass through a specific location every 15 minutes: https://www.mass.gov/traffic-volume-and-classification-in-massachusetts
- Census Data: avialable through mapc.org. Provides population data every 10 years: https://datacommon.mapc.org/browser/datasets/262
- MAPC Education related Data: Includes Enrollment and Dropout data by town


## Week 4: Feb/11 - Feb/17
- Deliverable: None 

### Successes: 
Continued to look for different data sources and download data.So far I have:

Communities Zoning Spreadsheet: https://www.mass.gov/doc/3a-info-sheetunit-capacity

General Info about the law: https://www.mass.gov/info-details/multi-family-zoning-requirement-for-mbta-communities

Flies from this website above:
- MBTA Communities Community Categories and Capacity Calculations
- Compliance Status Sheet as of 2-10-25

Data Sources

Shape Files: MA website

Traffic Data: https://www.mass.gov/traffic-volume-and-classification-in-massachusetts
- Still need to ingest

Census Data: STILL NEED

MAPC Data Sources: https://datacommon.mapc.org/

Education Data: 
- Enrollment by School Year (Schools) 2005-2024
- Drop Out rates from 2007-2022
- MCAS Science Grade 10: 2015-2023
- MCAS Math Grade 10: 2005-2025

### Failures:
No failures this week

### Difficulties: 
No difficulties this week.

## Week 5: Feb/18 - Feb/24
- Deliverable: Competency Demo - 5-minute demonstration of something the you have used and learned towards the project 

### Successes: 
Worked to ingest & store almost all shapefiles associated with housing. Created a small demo of this data, showing existing multi family housing and single family housing for each town

### Failures:
None

### Difficulties: 
The shapefiles were difficult to ingest, and took a long time. I also tried to ingest MBTA commuter rail shapefiles, but was not able to complete this. 

## Week 6: Feb/25 - Mar/3
- Deliverable: None 

### Successes: 

### Failures:
Busy week so not much time to work on this project

### Difficulties: 

## Week 7: Mar/4 - Mar/10

### Successes: 
Downloaded the census data assoicated with Massachusetts towns which includes population estimates broken down by age group. The education data I found can also be used to identify the number of childeren enrolled at each school.

### Failures:
Was hoping to get more down, but ran out of time. In the next two weeks I plan to make a lot of progress on the design doc, data storage, and research on the model to use.

### Difficulties: 
Still having trouble ingesting the MassDot traffic data. I might try webscraping, but it seems like it will be too difficult to get manually.

## Week 8: Mar/11 - Mar/17
- Deliverable: Design Document - Elaborate on the “hows” to build the project.  This document should be long and fairly low-level.

### Successes: 
- Began work on the database design to store all of the data.

### Failures:
- None

### Difficulties: 
- None

## Week 9: Mar/18 - Mar/24

### Successes: 
- Almost finished database design. Continued work on traffic data must be done to complete the design, but all other tables are generally designed.
- Table creation and data ingestion for town_nameplate & community_classification tables.

### Failures:
- None

### Difficulties: 
- Still no luck with the traffic data. Next week I will possibly try web scraping. I also want to spend time researching the best way to anaylize the traffic data. 

## Week 10: Mar/25 - Mar/31

### Deliverable: 
- Finish ingesting/storing data:
- MBTA Data
- Schools Data
- Town Data: data associated with law, and towns affected
- Land Data: reorginize already ingested shapefiles to match chosen format

### Successes: 
- Worked on ingesting and storing the MBTA data
- Started working on the schools data


### Failures:
- Still have some minor updates needed associated with the the town and land data but this can be resolved after the design doc is finished

### Difficulties: 
- Igesting and storing the data is taking longer than expected and some tasks will remain open while writinf rhe design doc. My goal is to have all data cleaned and stored before the start of the second semester, so the entire semester is focused on working on the model and visualiztions.

## Week 11: April/1 - April/7

### Deliverable: 
- Finish school data
- Ingest traffic data
- Research best way to predict/analyze traffic data
- Begin working on design doc
- Population Data: Will take a little longer due to ZIP/ZICTA conversion

### Successes: 
- Worked on writing a web scraping script to get the traffic data. While I have not gotten all of the data yet, I have a script that works, and I have stored the napeplate data associated with traffic locations, and began ingesting the traffic count data
- Finished ingesting and storing the mbta data. This includes the mbta stop napelate info, and the mbta trip data

To view the table creation or ingestion scripts, checkout:
- Database Info: database_set_up/table_creations_scripts & database_set_up/table_definitions
- Data Ingestions: data_ingestion/<data_type>

### Failures:

### Difficulties: 

## Week 12: April/8 - April/14

### Deliverable: 
- Continue researching models
- Continue working on design doc

In Person Meeting:
- Review progress on design doc

### Successes: 

### Failures:

### Difficulties: 

## Week 13: April/15 - April/21

### Deliverable: 
- Make edits to design doc based on general review in meeting

### Successes: 

### Failures:

### Difficulties: 

## Week 14: April/22 - April/28

### Deliverable: 
- Finalize design doc for review by the beginning of the summer session


### Successes: Design Doc

### Failures:

### Difficulties: 

## Week 15: April/29 - May/5

### Successes: 

### Failures:

### Difficulties: 

## Part 2

## Week 16: May/6 - May/12

### Successes: 

### Failures:

### Difficulties: 

## Week 17: May/13 - May/19

### Successes: 

### Failures:

### Difficulties: 