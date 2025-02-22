### SITE DESCRIPTION
Map of site

For this assignment we downloaded the Water Boundary Dataset for region 8 located in Mississippi (specifically watershed 080902030506). This watershed covers part of New Orleans and is near the Mississippi Delta. This is a good area to practice land classication since deltas are provide a diverse spread of land cover. The Mississippi Delta inclues uplands, the Gulf of Mexico, forests, mash, beaches, and urban areas. Please note that I was unable to figure out how to load in satilite imagrey under my watershed boundary so I used Open Street Map (OSM)

National Wildlife Federation. (n.d.). Mississippi River Delta. National Wildlife Federation. https://www.nwf.org/Educational-Resources/Wildlife-Guide/Wild-Places/Mississippi-River-Delta

### SILLOUETTE AND ELBOW PLOTS
In order to help determine the number of clusters I wanted to use in my K means calculation I used a combination of the Elbow Method and a Silloutee Score. For the Elbow Method I calculated and plotted the within-cluster sum of squares (WCSS) or inertia, as it is know in the Sklearn package, against different values of K, and then visually identified the "elbow" on the plot where the rate of decrease sharply changes and begins to flatten out, which is somewhere around 6 or 7 clusters. For the Silloutte Score I looked for values that were closer to one, which represented a more dense and well-seperated cluster. However, they seemed to trail off closer and closer to 1 - so I went with 7 clusters to in an attempt to get descreate land classification groups, while also trying not to over classify the data. 

Map of sillouettes and elbow plots

### RGB AND CLUSTER PLOTS
I plotted both the RGB color bands and the K means clusters to see how they visually compared. 
