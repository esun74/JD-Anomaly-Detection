#!/bin/bash
cat ./ag_od/test/sci.txt ./ag_od/test/sports.txt ./ag_od/test/world.txt > ./ag_od/test/business-outliers.txt
cat ./ag_od/test/business.txt ./ag_od/test/sports.txt ./ag_od/test/world.txt > ./ag_od/test/sci-outliers.txt
cat ./ag_od/test/business.txt ./ag_od/test/sci.txt ./ag_od/test/world.txt > ./ag_od/test/sports-outliers.txt
cat ./ag_od/test/business.txt ./ag_od/test/sci.txt ./ag_od/test/sports.txt > ./ag_od/test/world-outliers.txt