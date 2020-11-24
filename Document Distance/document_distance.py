# 6.0001 Spring 2020
# Problem Set 3
# Written by: sylvant, muneezap, charz, anabell, nhung, wang19k, asinelni, shahul, jcsands

# Problem Set 3
# Name: Yan Wu
# Collaborators: <insert collaborators>
# Time Spent: <insert time>
# Late Days Used: (only if you are using any)

import string

# - - - - - - - - - -
# Check for similarity by comparing two texts to see how similar they are to each other


### Problem 1: Prep Data ###
# Make a *small* change to separate the data by whitespace rather than just tabs
def load_file(filename):
    """
    Args:
        filename: string, name of file to read
    Returns:
        list of strings holding the file contents where
            each string was separated by a newline character (,) in the file
    """
    inFile = open(filename, 'r')
    line = inFile.read()
    inFile.close()
    line = line.strip().lower()
    for char in string.punctuation:
        line = line.replace(char, "")#remove all the punctuations
    return line.split()#split the individual words

### Problem 2: Find Ngrams ###
def find_ngrams(single_words, n):
    """
    Args:
        single_words: list of words in the text, in the order they appear in the text
            all words are made of lowercase characters
        n:            length of 'n-gram' window
    Returns:
        list of n-grams from input text list, or an empty list if n is not a valid value
    """
    n_gram = []
    if n > 0:
        for i in range(len(single_words)):
            if i+n <= len(single_words):#to make sure it doesn't go beyong the length of single_words
                n_seg = " ".join(single_words[i:i+n])
                n_gram.append(n_seg)
    return n_gram

### Problem 3: Word Frequency ###
def compute_frequencies(words):
    """
    Args:
        words: list of words (or n-grams), all are made of lowercase characters
    Returns:
        dictionary that maps string:int where each string
        is a word (or n-gram) in words and the corresponding int
        is the frequency of the word (or n-gram) in words
    """
    word_dict = {}
    for seg in words:
        if seg in word_dict:
            word_dict[seg] += 1#+1 count
        else:
            word_dict[seg] = 1#if seg isn't in the word_dict, create a new key&value
    return word_dict

### Problem 4: Similarity ###
def get_similarity_score(dict1, dict2, dissimilarity = False):
    """
    The keys of dict1 and dict2 are all lowercase,
    you will NOT need to worry about case sensitivity.

    Args:
        dict1: frequency dictionary of words or n-grams for one text
        dict2: frequency dictionary of words or n-grams for another text
        dissimilarity: Boolean, optional parameter. Default to False.
          If this is True, return the dissimilarity score, 100*(DIFF/ALL), instead.
    Returns:
        int, a percentage between 0 and 100, inclusive
        representing how similar the texts are to each other

        The difference in text frequencies = DIFF sums words
        from these three scenarios:
        * If a word or n-gram occurs in dict1 and dict2 then
          get the difference in frequencies
        * If a word or n-gram occurs only in dict1 then take the
          frequency from dict1
        * If a word or n-gram occurs only in dict2 then take the
          frequency from dict2
         The total frequencies = ALL is calculated by summing
         all frequencies in both dict1 and dict2.
        Return 100*(1-(DIFF/ALL)) rounded to the nearest whole number if dissimilarity
          is False, otherwise returns 100*(DIFF/ALL)
    """
    DIFF = 0
    ALL = sum(dict1.values())+sum(dict2.values())#summing all the frequencies in dict1 and dict2
    assert ALL != 0
    for i in dict2.keys():
        if i in dict1.keys():
            DIFF += abs(dict1[i] - dict2[i])#if i in both dict1 and dict2
        else:
            DIFF += dict2[i]#if i only in dict1
    for i in dict1.keys():
        if i not in dict2:
            DIFF += dict1[i]#if i only in dict2
    if dissimilarity == False:
        return round(100*(1-DIFF/ALL))
    else:
        return round(100*DIFF/ALL)

### Problem 5: Most Frequent Word(s) ###
def compute_most_frequent(dict1, dict2):
    """
    The keys of dict1 and dict2 are all lowercase,
    you will NOT need to worry about case sensitivity.

    Args:
        dict1: frequency dictionary for one text
        dict2: frequency dictionary for another text
    Returns:
        list of the most frequent word(s) in the input dictionaries

    The most frequent word:
        * is based on the combined word frequencies across both dictionaries.
          If a word occurs in both dictionaries, consider the sum the
          freqencies as the combined word frequency.
        * need not be in both dictionaries, i.e it can be exclusively in
          dict1, dict2, or shared by dict1 and dict2.
    If multiple words are tied (i.e. share the same highest frequency),
    return an alphabetically ordered list of all these words.
    """
    highest = []
    both = {}
    for i in dict1:
        if i in dict2:
            both[i]= dict1[i]+ dict2[i]#the dictionary where the words and freqencies in both dict1 and dict2
    both_highest = max(both.values())#the most frequent word when it is included in both dictionaries
    highest1 = max(dict1.values())#the most frequent word that's only in dict1
    highest2 = max(dict2.values())#the most frequent word that's only in dict2
    highestfreq = max(both_highest, highest1, highest2)
    for i in both: 
        if both[i] == highestfreq:
            highest.append(i)
    for i in dict1:
        if dict1[i] == highestfreq:
            highest.append(i)
    for i in dict2: 
        if dict2[i] == highestfreq:
            highest.append(i)
    #find the word with highest frequency in both, dict1, dict2
    highest.sort()#alphabetical order
    return highest
            

### Problem 6: Finding closest artist ###
def find_closest_artist(artist_to_songfiles, mystery_lyrics, ngrams = 1):
    """
    Args:
        artist_to_songfiles:
            dictionary that maps string:list of strings
            where each string key is an artist name
            and the corresponding list is a list of filenames (including the extension),
            each holding lyrics to a song by that artist
        mystery_lyrics: list of single word strings
            Can be more than one or two words (can also be an empty list)
            assume each string is made of lowercase characters
        ngrams: int, optional parameter. Default set to False.
            If it is greater than 1, n-grams of text in files
            and n-grams of mystery_lyrics should be used in analysis, with n
            set to the value of the parameter ngrams
    Returns:
        list of artists (in alphabetical order) that best match the mystery lyrics
        (i.e. list of artists that share the highest average similarity score (to the nearest whole number))

    The best match is defined as the artist(s) whose songs have the highest average
    similarity score (after rounding) with the mystery lyrics
    If there is only one such artist, then this function should return a singleton list
    containing only that artist.
    However, if all artists have an average similarity score of zero with respect to the
    mystery_lyrics, then this function should return an empty list. When no artists
    are included in the artist_to_songfiles, this function returns an empty list.
    """
    if len(artist_to_songfiles.keys())==0:#if there is no artist to begin with, there is no result
        return []
    song_dict = {}#initialize song dictionary
    
    for i in artist_to_songfiles.keys():
        song_dict[i]=[]
        for files in artist_to_songfiles[i]:
            word_lists = load_file(files)
            song_dict[i].append(word_lists)#loaded each file separately under each artist
            
    
    song_dict_freq = {}
    for i in song_dict.keys():
        song_dict_freq[i]=[]#the value needs to be a list to include all the dictionaries for individual songs
        for files in song_dict[i]:
            song_dict_freq[i].append(compute_frequencies(find_ngrams(files, ngrams)))#the frequency list of songs
    
    song_score = {}
    mys_freq = compute_frequencies(find_ngrams(mystery_lyrics, ngrams))#compute the frequency for the mystery_lyrics
    
    for i in song_dict_freq.keys():
        song_score[i] = []
        for dicts in song_dict_freq[i]:
            song_score[i].append(get_similarity_score(dicts, mys_freq))#similarity scores between freq of song files and mystery lyrics
    
    song_average = {}
    for i in song_score.keys():
        song_average[i] = sum(song_score[i])/len(song_score[i])#total scores/the number of scores
    
    artist=[]
    highest_score = max(song_average.values())
    
    for i in song_average.keys():
        if song_average[i] == highest_score and highest_score != 0:#if the highest score is 0, just return the empty list
            artist.append(i)
    
    return artist



if __name__ == "__main__":
    pass
    ##Uncomment the following lines to test your implementation
    ## Tests Problem 0: Prep Data
    test_directory = "tests/student_tests/"
    world, friend = load_file(test_directory + 'hello_world.txt'), load_file(test_directory + 'hello_friends.txt')
    print(world) ## should print ['hello', 'world', 'hello']
    print(friend) ## should print ['hello', 'friends']

    ## Tests Problem 1: Find Ngrams
    world_ngrams, friend_ngrams = find_ngrams(world, 2), find_ngrams(friend, 2)
    longer_ngrams = find_ngrams(world+world, 3)
    print(world_ngrams) ## should print ['hello world', 'world hello']
    print(friend_ngrams) ## should print ['hello friends']
    print(longer_ngrams) ## should print ['hello world hello', 'world hello hello', 'hello hello world', 'hello world hello']

    ## Tests Problem 2: Get frequency
    world_word_freq, world_ngram_freq = compute_frequencies(world), compute_frequencies(world_ngrams)
    friend_word_freq, friend_ngram_freq = compute_frequencies(friend), compute_frequencies(friend_ngrams)
    print(world_word_freq) ## should print {'hello': 2, 'world': 1}
    print(world_ngram_freq) ## should print {'hello world': 1, 'world hello': 1}
    print(friend_word_freq) ## should print {'hello': 1, 'friends': 1}
    print(friend_ngram_freq) ## should print {'hello friends': 1}

    ## Tests Problem 3: Similarity
    word_similarity = get_similarity_score(world_word_freq, friend_word_freq)
    ngram_similarity = get_similarity_score(world_ngram_freq, friend_ngram_freq)
    print(word_similarity) ## should print 40
    print(ngram_similarity) ## should print 0

    ## Tests Problem 4: Most Frequent Word(s)
    freq1, freq2 = {"hello":5, "world":1}, {"hello":1, "world":5}
    most_frequent = compute_most_frequent(freq1, freq2)
    print(most_frequent) ## should print ["hello", "world"]

    ## Tests Problem 5: Find closest matching artist
    test_directory = "tests/student_tests/"
    artist_to_songfiles_map = {
    "artist_1": [test_directory + "artist_1/song_1.txt", test_directory + "artist_1/song_2.txt", test_directory + "artist_1/song_3.txt"],
    "artist_2": [test_directory + "artist_2/song_1.txt", test_directory + "artist_2/song_2.txt", test_directory + "artist_2/song_3.txt"],
    }
    mystery_lyrics = load_file(test_directory + "mystery_lyrics/mystery_5.txt") # change which number mystery lyrics (1-5)
    print(find_closest_artist(artist_to_songfiles_map, mystery_lyrics)) # should print ['artist_1']
