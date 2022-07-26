#!/bin/bash
cat ./20ng_od/test/misc.txt ./20ng_od/test/pol.txt ./20ng_od/test/rec.txt ./20ng_od/test/rel.txt ./20ng_od/test/sci.txt > ./20ng_od/test/comp-outliers.txt
cat ./20ng_od/test/comp.txt ./20ng_od/test/pol.txt ./20ng_od/test/rec.txt ./20ng_od/test/rel.txt ./20ng_od/test/sci.txt > ./20ng_od/test/misc-outliers.txt
cat ./20ng_od/test/comp.txt ./20ng_od/test/misc.txt ./20ng_od/test/rec.txt ./20ng_od/test/rel.txt ./20ng_od/test/sci.txt > ./20ng_od/test/pol-outliers.txt
cat ./20ng_od/test/comp.txt ./20ng_od/test/misc.txt ./20ng_od/test/pol.txt ./20ng_od/test/rel.txt ./20ng_od/test/sci.txt > ./20ng_od/test/rec-outliers.txt
cat ./20ng_od/test/comp.txt ./20ng_od/test/misc.txt ./20ng_od/test/pol.txt ./20ng_od/test/rec.txt ./20ng_od/test/sci.txt > ./20ng_od/test/rel-outliers.txt
cat ./20ng_od/test/comp.txt ./20ng_od/test/misc.txt ./20ng_od/test/pol.txt ./20ng_od/test/rec.txt ./20ng_od/test/rel.txt > ./20ng_od/test/sci-outliers.txt
