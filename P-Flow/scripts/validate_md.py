#!/usr/bin/env python3
"""Cross-validation of M_d scores against ground-truth tags for all 200 samples."""

import csv
from collections import defaultdict

# Read md_scores
md = {}
with open('/Users/ztw/Desktop/videofake/P-Flow/data/md_scores.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        md[int(row['sample_id'])] = int(row['md_raw'])

# Read selected_200
gt = {}
with open('/Users/ztw/Desktop/videofake/P-Flow/data/selected_200.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        gt[int(row['id'])] = {
            'concept': row['concept'],
            'prompt': row['prompt'],
            'motion_level': row['motion_level']
        }

# ============================================================
# Define expected M_d for each sample based on prompt content
# ============================================================
# Rules:
# md=3 (object motion): animal locomoting, person walking/running/flying, vehicle moving,
#   ball thrown, collision between objects, acceleration of physical subject
# md=2 (camera/scene): FPV/drone/tracking/zoom, fluid dynamics, scene transformations,
#   deformation, smoke/steam, lighting changes
# md=1 (no-motion): truly static scenes

# Manual annotation of expected md for each sample ID based on prompt analysis
# Format: {sample_id: expected_md}

expected = {}

# Let's analyze each sample systematically
# I'll go through all 200 samples

# ID 7: physics - fluid dynamics, pirate ships sailing in coffee -> fluid + subject motion. Ships ARE sailing (locomoting) but in fluid. The primary motion is ships moving through fluid. The fluid dynamics is secondary. The prompt says "battling each other as they sail" - ships are locomoting subjects. -> md=3
expected[7] = 3

# ID 17: physics - acceleration, camera motion. SUV speeding up dirt road. Camera follows. Subject is accelerating. -> md=3
expected[17] = 3

# ID 21: unusual activity. Paper airplanes fluttering through jungle like migrating birds. Paper planes are objects moving/flying. -> md=3
expected[21] = 3

# ID 31: unusual activity. NYC submerged, fish/whales/sharks swim through streets. Animals locomoting. -> md=3
expected[31] = 3

# ID 32: animal. Golden retriever puppies playing in snow. Animals locomoting. -> md=3
expected[32] = 3

# ID 33: human - activity. Person running. -> md=3
expected[33] = 3

# ID 34: animal. Wolf pups frolicking and chasing. Animals locomoting. -> md=3
expected[34] = 3

# ID 43: animal, camera motion. Cat darting through garden, camera follows. Animal locomoting + camera. Primary = subject locomoting. -> md=3
expected[43] = 3

# ID 46: unusual subject. Giant cloud man shooting lightning. Cloud man is a subject but it "looms" (not locomoting) and "shoots lighting" - this is more scene/ambient transformation. The cloud is not walking/running. -> md=2
expected[46] = 2

# ID 47: animal. Samoyed and Golden Retriever romping through city. Animals locomoting. -> md=3
expected[47] = 3

# ID 50: unusual activity. White cat driving in a car. Cat is subject, car is moving. Subject locomoting (in a car). -> md=3
expected[50] = 3

# ID 70: scene. Subtle reflections of woman on train window, train moving at hyper-speed. This is scene/camera motion - the train moves, reflections change. No subject locomoting in frame. -> md=2
expected[70] = 2

# ID 72: scene, camera motion. FPV flying through underwater streets. Camera motion. -> md=2
expected[72] = 2

# ID 73: scene. Empty warehouse transformed by flora exploding from ground. Scene transformation. -> md=2
expected[73] = 2

# ID 74: unusual subject. Living flame wisp darting through fantasy market. "Darting" = locomoting. -> md=3
expected[74] = 3

# ID 76: scene, camera motion. FPV shot zooming through tunnel. Camera motion. -> md=2
expected[76] = 2

# ID 78: scene. Hyperlapse racing through tunnel into growing vines. Camera motion + scene transformation. -> md=2
expected[78] = 2

# ID 79: scene, camera motion. FPV internal locomotive cab of train. Camera/scene motion. -> md=2
expected[79] = 2

# ID 80: scene, camera motion. Zooming in hyper-fast to dandelion. Camera motion. -> md=2
expected[80] = 2

# ID 81: scene. Internal window of train at hyper-speed. Scene motion (view from train). -> md=2
expected[81] = 2

# ID 82: scene, camera motion. Handheld camera moving fast. Camera motion. -> md=2
expected[82] = 2

# ID 83: scene, camera motion. Super fast zoom out from mountain peak. Camera motion. -> md=2
expected[83] = 2

# ID 84: unusual activity, scene, camera motion. FPV shot flying through doors revealing waterfall. Camera + scene. The waterfall is fluid dynamics. Primary = camera/scene motion. -> md=2
expected[84] = 2

# ID 85: scene, camera motion. FPV shot rapidly flies towards house door. Camera motion. -> md=2
expected[85] = 2

# ID 88: scene. Tsunami coming through alley. Fluid dynamics / scene motion. -> md=2
expected[88] = 2

# ID 99: human - activity. Cloaked figure elevating in sky. Person flying/ascending. Subject locomoting. -> md=3
expected[99] = 3

# ID 104: unusual subject. Giant humanoid made of cotton candy stomping and roaring. Subject locomoting (stomping). -> md=3
expected[104] = 3

# ID 105: scene, camera motion. Zooming through dark forest with neon light. Camera motion. -> md=2
expected[105] = 2

# ID 106: scene. Cyclone of broken glass in alleyway. Fluid-like dynamics / scene motion (glass swirling like fluid). -> md=2
expected[106] = 2

# ID 111: scene, camera motion. Aerial shot of drone moving fast. Camera motion. -> md=2
expected[111] = 2

# ID 112: unusual activity. Hyperlapse through corridor + silver fabric flies through. Camera motion + object flying. The silver fabric flying through IS subject motion. -> md=3
expected[112] = 3

# ID 113: unusual activity. Aerial ocean, maelstrom/whirlpool forming. Fluid dynamics. -> md=2
expected[113] = 2

# ID 116: human - activity, camera motion. Woman running watching rocket. Person locomoting. -> md=3
expected[116] = 3

# ID 120: unusual activity. Pink pig running fast. Subject locomoting. -> md=3
expected[120] = 3

# ID 124: unusual activity, sequential motion. Lightning strikes turtle, turns into alligator. Scene transformation (morphing). The primary motion is transformation, not locomotion. -> md=2
expected[124] = 2

# ID 128: physics - acceleration. Vintage muscle cars drag racing. Subject accelerating. -> md=3
expected[128] = 3

# ID 135: unusual activity, unusual subject. Humans walking into dragon's jaws. People walking (locomoting). -> md=3
expected[135] = 3

# ID 136: scene. Police helicopter hovers above high-speed chase. Helicopter hovering + car chase. Cars are locomoting subjects. -> md=3
expected[136] = 3

# ID 139: human - activity. Futsal players showcasing skills. People moving. -> md=3
expected[139] = 3

# ID 143: unusual activity, sequential motion. Fish jumps out of tank and swims around head. Fish locomoting. -> md=3
expected[143] = 3

# ID 146: animal. Cat chasing mice. Animal locomoting. -> md=3
expected[146] = 3

# ID 156: human - activity. Synchronized diving. Person locomoting (diving). -> md=3
expected[156] = 3

# ID 157: unusual activity. Guitar swallowed by volcano, engulfed in magma. Scene transformation / fluid dynamics. No subject locomoting - guitar is being consumed. -> md=2
expected[157] = 2

# ID 159: physics - acceleration. School bus chugs up steep hill. Vehicle accelerating. -> md=3
expected[159] = 3

# ID 164: human - activity. Motorcycle stunt rider soars through air. Person + vehicle locomoting. -> md=3
expected[164] = 3

# ID 168: human - activity. Marathon runner crossing finish line. Person locomoting. -> md=3
expected[168] = 3

# ID 170: unusual activity. Penguin flies into mouth of blue whale. Penguin locomoting (flying). -> md=3
expected[170] = 3

# ID 172: human - activity. Girl running through forest with animals chasing. Person locomoting. -> md=3
expected[172] = 3

# ID 175: animal. Cat jumps onto kitchen counter. Animal locomoting. -> md=3
expected[175] = 3

# ID 177: human - activity. Skateboarders performing tricks. People locomoting. -> md=3
expected[177] = 3

# ID 178: animal. Ferret tosses ball, puppy chases. Animals locomoting. -> md=3
expected[178] = 3

# ID 183: human - activity. Soccer goalie diving save. Person locomoting. -> md=3
expected[183] = 3

# ID 184: human - activity. Bulldozer clears debris. Vehicle moving. -> md=3
expected[184] = 3

# ID 186: animal. Cat leaps out of box into taller box. Animal locomoting. -> md=3
expected[186] = 3

# ID 193: physics - gravity. Truck rolling backwards down hill. Vehicle moving (gravity). -> md=3
expected[193] = 3

# ID 195: human - activity. Person on uneven bars in gymnastics. Person moving. -> md=3
expected[195] = 3

# ID 197: human - activity. Girl grows wings on feet, soars across North America. Person flying. -> md=3
expected[197] = 3

# ID 202: physics - acceleration. Silver sedan glides around sharp corner. Vehicle accelerating. -> md=3
expected[202] = 3

# ID 228: human - emotion. Person's cheeks flushed, tripped in public. Person tripped = locomoting. -> md=3
expected[228] = 3

# ID 243: human - emotion. Person jumps up and down, expressing happiness. Person locomoting. -> md=3
expected[243] = 3

# ID 244: human - emotion. Close-up face, fear, navigating ship through storm. Close-up face with no full-body locomotion visible, just facial expression + implied ship movement. The visible motion is the face + storm scene. This is ambiguous but close-up face suggests md=2 (scene ambient). -> md=2
expected[244] = 2

# ID 247: camera motion. Static camera. Dinosaur running near lions. Subject locomoting (dinosaur running). Camera is static. -> md=3
expected[247] = 3

# ID 258: unusual activity. Garbage truck floating and spinning, defying gravity. The truck is a subject that's moving (spinning/floating). -> md=3
expected[258] = 3

# ID 260: human - activity, camera motion. FPV tracking shot of soccer player's feet dribbling. Person locomoting + camera tracking. Primary = subject locomoting. -> md=3
expected[260] = 3

# ID 262: physics - collision. Hands squeezing water ball, causing it to burst. This is a fluid dynamics event (water ball bursting), not a collision between solid objects. The burst releases fluid. -> md=2
expected[262] = 2

# ID 272: physics - acceleration. Female athlete sprints ahead. Person accelerating. -> md=3
expected[272] = 3

# ID 278: unusual activity, sequential motion. Bunny puts moon on back and flies into distance. Subject locomoting (bunny flying). -> md=3
expected[278] = 3

# ID 280: unusual activity. Man shows woman how to fold street upwards at 90 degrees. Scene transformation (buildings bending). People walking too, but the primary visual is the surreal transformation. However, people ARE walking through the street. The prompt says "man and woman are walking through a city street" - they are locomoting subjects. The street folding is additional. -> md=3
expected[280] = 3

# ID 283: physics - collision. Two basketballs thrown towards each other and collide. Object collision. -> md=3
expected[283] = 3

# ID 285: unusual activity. Skyscrapers transform into Gundam robot. Scene transformation (morphing). The robot itself then moves, but primary visual = morphing. However, a "moving Gundam robot" is locomoting... It says "transform into a moving Gundam robot" - so it transforms AND then moves. The transformation is the primary spectacle. -> md=2
expected[285] = 2

# ID 289: scene, camera motion. Drone view zooming into closet, other end opens to reveal pyramid world. Camera motion + scene transformation. -> md=2
expected[289] = 2

# ID 290: scene. Rollercoaster ride from city to desert to ice world. Camera/scene motion (riding rollercoaster). -> md=2
expected[290] = 2

# ID 294: human - activity. Asian girl Hip-Hop dancing. Person locomoting. -> md=3
expected[294] = 3

# ID 297: scene. Female warrior rushes towards camera, then turns into holographic monster. Subject locomoting (rushing) then transforms. The rushing IS locomotion. -> md=3
expected[297] = 3

# ID 340: human - emotion. Close-up man's face, muscles tensed, breathing heavily. Close-up face with no full-body locomotion. The "hyperspeed, dynamic motion, fiery" is stylistic. -> md=2
expected[340] = 2

# ID 341: physics - collision. Two cars colliding at intersection. Object collision. -> md=3
expected[341] = 3

# ID 343: physics - collision. Two football players colliding. Object/subject collision. -> md=3
expected[343] = 3

# ID 344: physics - collision. Meteor colliding with planet. Object collision. -> md=3
expected[344] = 3

# ID 345: physics - collision. Skateboarder losing control, colliding with bench. Subject + object collision. -> md=3
expected[345] = 3

# ID 346: physics - collision, human - activity, camera motion. Ping-pong game, rapid back-and-forth of ball. Ball collision + camera zoom. The ball is a subject colliding. -> md=3
expected[346] = 3

# ID 347: physics - collision. Bird flying into glass window. Subject collision. -> md=3
expected[347] = 3

# ID 348: physics - collision. Shopping cart rolling down hill, colliding with parked car. Object collision. -> md=3
expected[348] = 3

# ID 350: physics - fluid dynamics. Raindrops hitting puddle, ripples and splashes. Fluid dynamics. -> md=2
expected[350] = 2

# ID 351: physics - fluid dynamics. Water jet cutting through metal. Fluid dynamics. -> md=2
expected[351] = 2

# ID 353: physics - fluid dynamics. Water balloon bursting, water forming sphere. Fluid dynamics. -> md=2
expected[353] = 2

# ID 355: physics - fluid dynamics. Waterfall, water crashing down. Fluid dynamics. -> md=2
expected[355] = 2

# ID 356: physics - fluid dynamics. Soap bubble popping, liquid dispersing. Fluid dynamics. -> md=2
expected[356] = 2

# ID 359: physics - acceleration. Runner accelerating up hill. Person accelerating. -> md=3
expected[359] = 3

# ID 361: physics - acceleration. Speedboat accelerating across lake. Vehicle accelerating. -> md=3
expected[361] = 3

# ID 362: physics - acceleration. Horse accelerating out of starting gate. Animal accelerating. -> md=3
expected[362] = 3

# ID 363: physics - gravity, acceleration. Rocket blasting off, accelerating. Subject accelerating. -> md=3
expected[363] = 3

# ID 367: physics - gravity. Meteor entering atmosphere, falling. Subject moving (falling). -> md=3
expected[367] = 3

# ID 370: unusual activity. Man transforming into superhero in forest. Scene transformation (morphing). -> md=2
expected[370] = 2

# ID 373: unusual activity. Man running in forest, transforms into wolf. Person locomoting + transformation. The running IS locomotion. -> md=3
expected[373] = 3

# ID 382: animal. Tabby cat darting across back street alley. Animal locomoting. -> md=3
expected[382] = 3

# ID 397: human - activity. Man BASE jumping, macaw flies alongside. Person locomoting (falling/flying). -> md=3
expected[397] = 3

# ID 402: animal, unusual activity. Puppies exploring ruins in the sky. Animals locomoting. -> md=3
expected[402] = 3

# ID 406: unusual activity. Whirlwind of colorful fabrics fluttering and swirling. Scene/ambient motion (fabrics swirling like fluid). No discrete subject locomoting. -> md=2
expected[406] = 2

# ID 407: unusual activity. Lamp transforming into flamingo. Scene transformation (morphing). Camera circles around. -> md=2
expected[407] = 2

# ID 408: unusual activity. Broom morphing into peacock. Scene transformation (morphing). -> md=2
expected[408] = 2

# ID 409: unusual activity. Plant transforming into octopus. Scene transformation (morphing). -> md=2
expected[409] = 2

# ID 419: physics - collision. Marble goes through glass cup, breaking it. Object collision. -> md=3
expected[419] = 3

# ID 431: human - activity. First-person view of running upstairs. Person locomoting. -> md=3
expected[431] = 3

# ID 454: physics - fluid dynamics. Whirlpool in river. Fluid dynamics. -> md=2
expected[454] = 2

# ID 455: physics - fluid dynamics. Champagne being poured into glass, bubbles rising. Fluid dynamics. -> md=2
expected[455] = 2

# ID 456: physics - fluid dynamics. Liquid droplet bouncing on water-repellent surface. Fluid dynamics. -> md=2
expected[456] = 2

# ID 458: physics - fluid dynamics. Fountain, water shooting upwards. Fluid dynamics. -> md=2
expected[458] = 2

# ID 461: physics - fluid dynamics. Drink being stirred, swirling liquid. Fluid dynamics. -> md=2
expected[461] = 2

# ID 466: physics - fluid dynamics. Syringe injecting liquid into vial. Fluid dynamics. -> md=2
expected[466] = 2

# ID 468: physics - fluid dynamics. Splash created by stone in pond. Fluid dynamics. -> md=2
expected[468] = 2

# ID 471: physics - fluid dynamics. Whirlpool forming in sink. Fluid dynamics. -> md=2
expected[471] = 2

# ID 474: physics - fluid dynamics. River rapid, turbulent water. Fluid dynamics. -> md=2
expected[474] = 2

# ID 475: physics - fluid dynamics. Water-filled balloon sliced open, water flowing out. Fluid dynamics. -> md=2
expected[475] = 2

# ID 477: physics - fluid dynamics. Beverage can opened, spray and bubbles. Fluid dynamics. -> md=2
expected[477] = 2

# ID 479: physics - fluid dynamics. Liquid droplet forming and falling from faucet. Fluid dynamics. -> md=2
expected[479] = 2

# ID 491: scene. Flying cars zoom through futuristic cityscape. Subject locomoting (cars flying). -> md=3
expected[491] = 3

# ID 526: scene. Ships taking off and landing at spaceport. Subject locomoting (ships). -> md=3
expected[526] = 3

# ID 555: unusual subject. Robots in snowball fight, throwing and dodging. Subject locomoting. -> md=3
expected[555] = 3

# ID 556: unusual activity. Characters from paintings step out of frames, throwing snowballs. Subject locomoting. -> md=3
expected[556] = 3

# ID 557: human - activity. Couple runs through downpour, splashing in puddles. People locomoting. -> md=3
expected[557] = 3

# ID 560: animal, unusual activity. Squirrel piloting miniature airplane. Subject moving (airplane flying). -> md=3
expected[560] = 3

# ID 569: animal, unusual activity. Bear flying through sky. Subject locomoting. -> md=3
expected[569] = 3

# ID 575: animal, unusual activity. Butterfly in race car speeding around track. Subject moving. -> md=3
expected[575] = 3

# ID 577: animal, unusual activity. Fox steering ship through stormy sea. Subject moving (ship). -> md=3
expected[577] = 3

# ID 578: animal, unusual activity. Turtle riding skateboard down hill. Subject locomoting. -> md=3
expected[578] = 3

# ID 580: animal, unusual activity. Kangaroo sparring with punching bag. Subject locomoting. -> md=3
expected[580] = 3

# ID 590: human - activity, unusual activity. Person riding bicycle on tightrope. Person locomoting. -> md=3
expected[590] = 3

# ID 591: human - activity, unusual activity. Person swimming through air. Person locomoting. -> md=3
expected[591] = 3

# ID 605: human - activity, unusual activity. Person flying kite made of fire. Person moving + object moving. -> md=3
expected[605] = 3

# ID 610: human - activity, unusual activity. Person juggling planets. Person moving. -> md=3
expected[610] = 3

# ID 612: human - activity, unusual activity. Person painting graffiti on flying spaceship. Person moving (painting action). -> md=3
expected[612] = 3

# ID 617: human - activity, unusual activity. Person playing guitar made of lightning. Person moving. -> md=3
expected[617] = 3

# ID 627: human - activity, unusual activity. Person running on treadmill through dimensions. Person locomoting. -> md=3
expected[627] = 3

# ID 629: human - activity, unusual activity. Person diving into pool of liquid crystal. Person locomoting. -> md=3
expected[629] = 3

# ID 633: human - activity, unusual activity. Person skydiving from hot air balloon. Person locomoting. -> md=3
expected[633] = 3

# ID 636: human - activity, unusual activity. Person playing drum set of thunderclouds. Person moving. -> md=3
expected[636] = 3

# ID 641: scene, unusual activity. Pouring milk into bowl that transitions to vast ocean with whale. Scene transformation (milk → ocean) + fluid dynamics. The whale being thrown by waves is subject motion but the primary spectacle is the transformation. However whale IS being thrown around = subject locomoting. -> md=3
expected[641] = 3

# ID 642: physics - collision. Dog colliding with cat, both tumbling. Subject collision. -> md=3
expected[642] = 3

# ID 643: physics - collision. Person on Segway colliding with pedestrian. Subject collision. -> md=3
expected[643] = 3

# ID 645: physics - collision. Cyclist colliding with stop sign. Subject/object collision. -> md=3
expected[645] = 3

# ID 646: physics - collision. Two RC planes colliding mid-air. Object collision. -> md=3
expected[646] = 3

# ID 647: physics - collision. Person walking colliding with lamppost. Subject collision. -> md=3
expected[647] = 3

# ID 648: physics - collision. Skateboarder colliding with curb. Subject collision. -> md=3
expected[648] = 3

# ID 650: physics - collision. Two people on roller skates colliding. Subject collision. -> md=3
expected[650] = 3

# ID 651: physics - collision. Person on hoverboard colliding with wall. Subject collision. -> md=3
expected[651] = 3

# ID 652: physics - collision. Two boats colliding in marina. Object collision. -> md=3
expected[652] = 3

# ID 653: physics - collision. Person on scooter colliding with park bench. Subject collision. -> md=3
expected[653] = 3

# ID 654: physics - acceleration. Skateboarder accelerating down steep hill. Subject accelerating. -> md=3
expected[654] = 3

# ID 655: physics - acceleration. Cheetah accelerating to full speed. Animal accelerating. -> md=3
expected[655] = 3

# ID 656: physics - acceleration. High-speed train accelerating. Vehicle accelerating. -> md=3
expected[656] = 3

# ID 657: physics - acceleration. Spaceship entering hyperdrive. Subject accelerating. -> md=3
expected[657] = 3

# ID 658: physics - acceleration. Drag racer accelerating. Vehicle accelerating. -> md=3
expected[658] = 3

# ID 659: physics - acceleration. Sports car accelerating. Vehicle accelerating. -> md=3
expected[659] = 3

# ID 660: physics - acceleration. Jet fighter accelerating off carrier. Vehicle accelerating. -> md=3
expected[660] = 3

# ID 661: physics - acceleration. Speedboat accelerating across lake. Vehicle accelerating. -> md=3
expected[661] = 3

# ID 662: physics - acceleration. Skier accelerating down slope. Person accelerating. -> md=3
expected[662] = 3

# ID 663: physics - acceleration. Drone accelerating through forest. Vehicle accelerating. -> md=3
expected[663] = 3

# ID 664: physics - acceleration. Horse accelerating out of gate. Animal accelerating. -> md=3
expected[664] = 3

# ID 665: physics - acceleration. Dog accelerating after being let off leash. Animal accelerating. -> md=3
expected[665] = 3

# ID 666: physics - acceleration. Helicopter accelerating as it lifts off. Vehicle accelerating. -> md=3
expected[666] = 3

# ID 668: physics - acceleration. Jet ski accelerating across water. Vehicle accelerating. -> md=3
expected[668] = 3

# ID 669: physics - acceleration. Racehorse accelerating on final stretch. Animal accelerating. -> md=3
expected[669] = 3

# ID 671: physics - acceleration. Base jumper accelerating after leaping off cliff. Person accelerating. -> md=3
expected[671] = 3

# ID 672: physics - acceleration. Cyclist accelerating out of saddle. Person accelerating. -> md=3
expected[672] = 3

# ID 673: physics - acceleration. Longboarder accelerating downhill. Person accelerating. -> md=3
expected[673] = 3

# ID 674: physics - acceleration. Skydiver accelerating during free fall. Person accelerating. -> md=3
expected[674] = 3

# ID 675: physics - acceleration. Motocross bike accelerating. Vehicle accelerating. -> md=3
expected[675] = 3

# ID 677: physics - acceleration. Snowboarder accelerating down slope. Person accelerating. -> md=3
expected[677] = 3

# ID 678: physics - acceleration. Race car accelerating through chicane. Vehicle accelerating. -> md=3
expected[678] = 3

# ID 679: physics - acceleration. Surfer accelerating on wave. Person accelerating. -> md=3
expected[679] = 3

# ID 692: unusual activity. Man wearing diving helmet with jetpack walking on lava, dragon flies behind. Subject locomoting (man walking, dragon flying). -> md=3
expected[692] = 3

# ID 695: human - activity. Tracking camera FPV, scooter zooms through supermarket. Person locomoting + camera. Primary = subject locomoting. -> md=3
expected[695] = 3

# ID 700: physics - deformation. Rubber band stretched then released. Deformation. -> md=2
expected[700] = 2

# ID 701: physics - deformation. Metal spring compressed then released. Deformation. -> md=2
expected[701] = 2

# ID 704: physics - deformation. Trampoline surface bending under weight. Deformation. -> md=2
expected[704] = 2

# ID 706: physics - deformation. Elastic fabric pulled and stretched. Deformation. -> md=2
expected[706] = 2

# ID 707: physics - deformation. Plastic ruler bent then snaps back. Deformation. -> md=2
expected[707] = 2

# ID 729: physics - thermodynamics. Water droplet on hot surface, evaporating into steam. Thermodynamics / ambient. -> md=2
expected[729] = 2

# ID 735: physics - thermodynamics. Water balloon popped, liquid maintains shape then cascades. Fluid dynamics + thermodynamics. -> md=2
expected[735] = 2

# ID 756: camera motion, human - activity. Low-angle shot of dancer leaping. Person locomoting + camera. Primary = subject locomoting. -> md=3
expected[756] = 3

# ID 759: camera motion, human - activity. FPV of cyclist riding through city. Person locomoting + camera. Primary = subject locomoting. -> md=3
expected[759] = 3

# ID 761: camera motion, human - activity. FPV of surfer paddling out and catching wave. Person locomoting + camera. Primary = subject locomoting. -> md=3
expected[761] = 3

# ID 780: camera motion, scene. Aerial shot of city intersection at rush hour. Camera/scene motion. No single subject locomoting (cars move but the shot is about the overall scene). -> md=2
expected[780] = 2

# ID 792: camera motion, scene. Truck alongside train moving through countryside. Camera/scene motion (tracking). Both vehicles move but the shot is about the changing landscape. -> md=2
expected[792] = 2

# ID 800: camera motion, scene. Truck through bustling street market. Camera/scene motion. -> md=2
expected[800] = 2

# ID 802: camera motion, scene. Truck through tranquil garden. Camera/scene motion. -> md=2
expected[802] = 2

# ID 829: camera motion, scene. Push-in through crowd at festival towards performer. Camera motion. -> md=2
expected[829] = 2

# ID 844: camera motion, scene. Handheld shot following child running through field. Camera following subject. Subject IS locomoting. -> md=3
expected[844] = 3

# ID 848: camera motion, scene. Handheld camera following dog running through park. Subject IS locomoting. -> md=3
expected[848] = 3

# ID 849: camera motion, human - activity. Tracking shot of skateboarder performing tricks. Subject IS locomoting. -> md=3
expected[849] = 3

# ID 850: camera motion, scene. Tracking shot of car driving along mountain road. Subject IS locomoting (car driving). -> md=3
expected[850] = 3

# ID 852: camera motion, human - activity. Tracking shot of cyclists racing through forest. Subjects ARE locomoting. -> md=3
expected[852] = 3

# ID 853: camera motion, scene. Tracking shot of train through snowy landscape. Subject IS locomoting (train). -> md=3
expected[853] = 3

# ID 857: unusual subject. Tracking shot of gremlins on rollercoaster. Subjects ARE locomoting (on rollercoaster). -> md=3
expected[857] = 3

# ID 858: unusual activity. Tracking shot. Scuba diver runs down busy street. Subject IS locomoting. -> md=3
expected[858] = 3

# ID 859: unusual subject, camera motion. Camera tracking shot. Gigantic flying monster flies through city. Subject IS locomoting (flying). -> md=3
expected[859] = 3

# ID 862: unusual subject, camera motion. Over shoulder shot. Lizard creature pushing buttons in giant robot stomping through city. Subject IS acting (pushing buttons) + robot stomping = locomoting. -> md=3
expected[862] = 3

# ID 867: scene. Slow-motion volcanic landscape, lava spewing, camera flies through. Fluid dynamics / scene + camera. -> md=2
expected[867] = 2

# ID 873: unusual activity. Vintage rocket man on spaceship flying through blood vessel. Subject IS locomoting (flying). -> md=3
expected[873] = 3

# ID 879: human - activity. Tracking camera FPV, scooter zooms through supermarket. Person locomoting + camera. Primary = subject locomoting. -> md=3
expected[879] = 3

# ============================================================
# Now validate
# ============================================================

# Cross-tabulation
crosstab = defaultdict(lambda: defaultdict(int))
results = []
wrong_samples = defaultdict(list)

# Define tag categories from concept field
def get_primary_tag(concept):
    """Extract primary tag category from concept string."""
    concepts = [c.strip() for c in concept.split(',')]
    # Return the first/most specific concept
    return concepts[0]

def get_tag_category(concept):
    """Map concept to a higher-level category."""
    concepts = [c.strip() for c in concept.split(',')]
    # Classify into broader groups
    for c in concepts:
        if 'physics - fluid' in c:
            return 'physics-fluid'
        if 'physics - collision' in c:
            return 'physics-collision'
        if 'physics - deformation' in c:
            return 'physics-deformation'
        if 'physics - thermodynamics' in c:
            return 'physics-thermodynamics'
        if 'physics - acceleration' in c:
            return 'physics-acceleration'
        if 'physics - gravity' in c:
            return 'physics-gravity'
    # If no physics tag, use the first concept
    first = concepts[0]
    if first == 'animal':
        return 'animal'
    if first.startswith('human'):
        return first  # human - activity, human - emotion
    if 'camera motion' in concepts and 'scene' in concepts:
        return 'camera+scene'
    if 'camera motion' in concepts:
        return 'camera-motion'
    if first == 'scene':
        return 'scene'
    if first == 'unusual activity' or first == 'unusual subject':
        # Check if there are other tags
        for c in concepts:
            if c != first:
                return f'{first}+{c}'
        return first
    if first == 'sequential motion':
        return 'sequential-motion'
    return first

for sid in sorted(gt.keys()):
    if sid not in md:
        continue
    if sid not in expected:
        continue
    
    md_raw = md[sid]
    exp = expected[sid]
    concept = gt[sid]['concept']
    prompt = gt[sid]['prompt']
    tag_cat = get_tag_category(concept)
    
    crosstab[tag_cat][md_raw] += 1
    
    correct = (md_raw == exp)
    verdict = '✅' if correct else '❌'
    
    results.append({
        'id': sid,
        'concept': concept,
        'tag_cat': tag_cat,
        'md_raw': md_raw,
        'expected': exp,
        'verdict': verdict,
        'prompt': prompt[:80]
    })
    
    if not correct:
        error_type = f'md={md_raw}→should_be={exp}'
        wrong_samples[error_type].append({
            'id': sid,
            'concept': concept,
            'tag_cat': tag_cat,
            'prompt': prompt[:120]
        })

# ============================================================
# OUTPUT
# ============================================================

print("=" * 100)
print("CROSS-VALIDATION OF M_d SCORES vs GROUND-TRUTH TAGS")
print("=" * 100)

# 1. Cross-tabulation
print("\n" + "=" * 80)
print("1. CROSS-TABULATION: tag category × md_raw")
print("=" * 80)
print(f"{'Tag Category':<30} {'md=1':>6} {'md=2':>6} {'md=3':>6} {'Total':>6}")
print("-" * 80)
for cat in sorted(crosstab.keys()):
    c1 = crosstab[cat][1]
    c2 = crosstab[cat][2]
    c3 = crosstab[cat][3]
    total = c1 + c2 + c3
    print(f"{cat:<30} {c1:>6} {c2:>6} {c3:>6} {total:>6}")

# Totals
total_by_md = defaultdict(int)
for cat in crosstab:
    for m, cnt in crosstab[cat].items():
        total_by_md[m] += cnt
print("-" * 80)
print(f"{'TOTAL':<30} {total_by_md[1]:>6} {total_by_md[2]:>6} {total_by_md[3]:>6} {sum(total_by_md.values()):>6}")

# 2. Per-sample results
print("\n" + "=" * 80)
print("2. PER-SAMPLE RESULTS (all 200)")
print("=" * 80)
print(f"{'ID':>5} {'Concept Tag':<45} {'md':>3} {'Exp':>3} {'OK':>3}")
print("-" * 80)
for r in results:
    print(f"{r['id']:>5} {r['concept'][:45]:<45} {r['md_raw']:>3} {r['expected']:>3} {r['verdict']:>3}")

# 3. Summary
correct_count = sum(1 for r in results if r['verdict'] == '✅')
wrong_count = sum(1 for r in results if r['verdict'] == '❌')
total = len(results)

print("\n" + "=" * 80)
print("3. SUMMARY")
print("=" * 80)
print(f"Total samples: {total}")
print(f"Correct:  {correct_count} ({100*correct_count/total:.1f}%)")
print(f"Wrong:    {wrong_count} ({100*wrong_count/total:.1f}%)")

# Wrong by category
wrong_by_cat = defaultdict(int)
for r in results:
    if r['verdict'] == '❌':
        wrong_by_cat[r['tag_cat']] += 1

print("\nWrong by tag category:")
for cat in sorted(wrong_by_cat.keys()):
    # Get total for this category
    cat_total = sum(crosstab[cat].values())
    print(f"  {cat:<30} {wrong_by_cat[cat]:>3} wrong / {cat_total:>3} total ({100*wrong_by_cat[cat]/cat_total:.1f}%)")

# 4. All wrong samples grouped by error type
print("\n" + "=" * 80)
print("4. WRONG SAMPLES BY ERROR TYPE")
print("=" * 80)
for error_type in sorted(wrong_samples.keys()):
    samples = wrong_samples[error_type]
    print(f"\n--- {error_type} ({len(samples)} samples) ---")
    for s in samples:
        print(f"  ID={s['id']:>4} [{s['tag_cat']}] {s['prompt']}")
