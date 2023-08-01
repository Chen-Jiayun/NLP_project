import skip_gram_model

def skip_gram(window_size=1):
   skip_grams = []
   c = skip_gram_model.corpus
   for i in range(window_size, len(c) - window_size):
       target = [i]
       context = [c[i - window_size], c[i + window_size]]
       for w in context:
           skip_grams.append([target, w])

   return skip_grams

skip_gram_list = skip_gram()

print(skip_gram_list)