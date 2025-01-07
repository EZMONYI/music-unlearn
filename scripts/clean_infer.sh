sh infer_lyric.sh | grep ^H | cut -f1,2,3 > ../results/lyric.inf
sh infer_lyric.sh | grep ^T | cut -f1,2 > ../results/lyric.ref
sh infer_melody.sh | grep ^H | cut -f1,2,3 > ../results/melody.inf
sh infer_melody.sh | grep ^T | cut -f1,2 > ../results/melody.ref
