import json

nocontext = "generation.one_keyword.json"
#prepared_predicted_context = "generation.prepared_predicted_context.json"
#on_the_fly_predicted_context = "generation.on_the_fly_predicted_context.json"

def get_json(json_file):
    with open(json_file) as f:
        return json.load(f)

nocontext = get_json(nocontext)
#prepared_predicted_context = get_json(prepared_predicted_context)
#on_the_fly_predicted_context = get_json(on_the_fly_predicted_context)

for video in nocontext:
    print(nocontext[video]['clip_order'])
    for clip_name in nocontext[video]['clip_order']:
        if clip_name in nocontext[video]:
            print(video + "      " + clip_name)
            #print(video + "-" + str(nocontext[video]['clip_order'].index(clip_name)))
            if 'prompt' in nocontext[video][clip_name]:
                print("[prompt]:", nocontext[video][clip_name]['prompt'])
            if 'reference' in nocontext[video][clip_name]:
                print("[ref]:", nocontext[video][clip_name]['reference'])
            else:
                print("[ref]:", nocontext[video][clip_name]['translation'])
            print("[nocontext]:", nocontext[video][clip_name]['hypothesis'])
            #print("[prepared_context]:", prepared_predicted_context[video][clip_name]['hypothesis'])
            #print("[on_the_fly_context]:", on_the_fly_predicted_context[video][clip_name]['hypothesis'])
            print("\n")