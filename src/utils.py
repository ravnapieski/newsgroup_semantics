import os
import tarfile
import urllib.request
import numpy
import nltk
from nltk.stem import WordNetLemmatizer

def download_dataset(url, archive_path, target_folder):
    """
    Downloads and extracts the dataset.
    
    Args:
        url (str): The URL to download from.
        archive_path (str): The full local path to save the .tar.gz file (e.g., ./data/file.tar.gz).
        target_folder (str): The full path where the extracted folder should end up (e.g., ./data/20_newsgroups).
    """
    # check if the final folder already exists
    if not os.path.exists(target_folder):
        print(f"Dataset not found at {target_folder}.")
        
        # create the parent directory (./data) if it doesn't exist
        parent_dir = os.path.dirname(archive_path)
        if not os.path.exists(parent_dir):
            print(f"Creating directory: {parent_dir}")
            os.makedirs(parent_dir, exist_ok=True)

        # download
        if not os.path.exists(archive_path):
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, archive_path)
            print("Download complete.")
        
        print("Extracting...")
        
        # extract
        with tarfile.open(archive_path, "r:gz") as tar:
            
            # safe extract
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tar, path=parent_dir)
            
        print(f"Extraction complete. Data is in {target_folder}")
    else:
        print(f"Dataset already exists at {target_folder}.")
    
def load_newsgroups(base_folder, target_newsgroups):
    documents = []
    doc_filenames = []
    doc_labels = []
    
    for dirpath, dirnames, filenames in os.walk(base_folder):
        current_newsgroup = os.path.basename(dirpath)
        
        if current_newsgroup in target_newsgroups:
            print(f"Processing documents in: {current_newsgroup}")
            # Sort filenames to ensure consistent order every time you run it
            filenames.sort() 
            
            count = 0
            for filename in filenames:
                full_file_path = os.path.join(dirpath, filename)
                count += 1
                
                # open file
                with open(full_file_path, 'r', encoding='latin-1') as f:
                    document = f.read()
                    documents.append(document)
                    
                    identifier = os.path.join(current_newsgroup, filename)
                    doc_filenames.append(identifier)
                    
                    doc_labels.append(current_newsgroup)
                    
            print(f"Processed {count} files.")
            
    return documents, doc_filenames, doc_labels

def prune_vocabulary(unified_vocabulary, unified_indices):
    
    stopwords = set(nltk.corpus.stopwords.words('english'))
    # Get word counts
    total_counts = numpy.bincount(unified_indices)
    active_counts = total_counts[total_counts > 0]
    
    overly_frequent_threshold = numpy.percentile(active_counts, 99)
    print(f"Total unique words before pruning: {len(unified_vocabulary)}")
    print(f"Count threshold for top 1%: {overly_frequent_threshold:.0f} occurrences")
    overly_infrequent_threshold = numpy.percentile(active_counts, 35)
    print(f"Count threshold for bottom 35%: {overly_infrequent_threshold:.0f} occurrences")
    
    pruningdecisions = numpy.zeros((len(unified_vocabulary),1))
    
    # Remove...
    for i, word in enumerate(unified_vocabulary):
        # 1. Stopwords
        if (word in stopwords):
            pruningdecisions[i] = True
        # 2. Overly frequent words (either overall frequency or document
        # frequency). E.g. leave out the top 1% most frequent words.
        # Here we use overall frequency (relative, not absolute)
        elif total_counts[i] > overly_frequent_threshold:
            pruningdecisions[i] = True
        # 3. Overly infrequent words (either overall frequency or
        # document frequency). E.g. leave out the bottom 35% most
        # infrequent words.
        elif total_counts[i] <= overly_infrequent_threshold:
            pruningdecisions[i] = True
        # 4. Overly short words (e.g. single-character words)
        elif len(word) < 2:
            pruningdecisions[i] = True
        # 5. Overly long words over 20 characters
        elif len(word) > 20:
            pruningdecisions[i] = True
        #  6. Words that contain unwanted characters, e.g. numbers or
        # special characters (but accented characters are ok)
        elif word.isalpha() == False:
            pruningdecisions[i] = True
    
    #Final pruned vocabulary
    pruned_vocab = [word for i, word in enumerate(unified_vocabulary) if not pruningdecisions[i]]
    print(f"Total unique words after pruning: {len(pruned_vocab)}")

    # Build a mapping from old index -> new index
    old_to_new = {}
    new_idx = 0
    for old_idx, remove in enumerate(pruningdecisions):
        if not remove:
            old_to_new[old_idx] = new_idx
            new_idx += 1
    if not old_to_new: # Handle case where all words are pruned
         print("Warning: All words were pruned.")
         return numpy.array(pruned_vocab), numpy.array([])

    # Remap unified_indices to new pruned indices, ignoring removed words (+ keeping only valid ones)
    pruned_indices = numpy.array([old_to_new[i] for i in unified_indices if i in old_to_new])
    return pruned_vocab, pruned_indices

def tagtowordnet(postag):
    if postag.startswith('N'): return 'n'
    if postag.startswith('V'): return 'v'
    if postag.startswith('J'): return 'a'
    if postag.startswith('R'): return 'r'
    return -1

def lemmatizetext(nltk_tokens, lemmatizer):
    tagged_text = nltk.pos_tag(nltk_tokens)
    lemmatized_text = []
    for word, tag in tagged_text:
        wordnet_tag = tagtowordnet(tag)
        if wordnet_tag != -1:
            lemmatized_word = lemmatizer.lemmatize(word, wordnet_tag)
        else:
            lemmatized_word = word
        lemmatized_text.append(lemmatized_word)
    return lemmatized_text

def process_documents(documents):
    
    print("Processing documents...")
    # Init lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Create a unified list of all words to build the vocabulary
    all_lemmatized_words = []
    
    corpus_lemmatized = []
    
    for document in documents:
        # tokenize
        tokens = nltk.word_tokenize(document)
        nltk_text = nltk.Text(tokens)
        # make lowercase
        tokens_lower  = [w.lower() for w in tokens]
        
        # lemmatize
        lemmatized_tokens = lemmatizetext(tokens_lower, lemmatizer)
        
        # Store the lemmatized tokens for this document
        corpus_lemmatized.append(lemmatized_tokens)
        # gather all lemmatized words
        all_lemmatized_words.extend(lemmatized_tokens)
        
    
    # Create unified vocabulary
    print("Creating unified vocabulary...")
    unified_vocabulary, unified_indices = numpy.unique(all_lemmatized_words, return_inverse=True)

    # Prune vocabulary
    print("Pruning vocabulary...")
    vocab, vocab_indices = prune_vocabulary(unified_vocabulary, unified_indices)
    
    # Create mycrawled_prunedtexts
    print("Creating pruned document list...")
    
    # set for fast O(1) lookups
    pruned_vocab_set = set(vocab)
    
    pruned_documents = []
    
    # Loop through the lemmatized texts and keep only words in the pruned set
    for doc_tokens in corpus_lemmatized:
        pruned_doc = [word for word in doc_tokens if word in pruned_vocab_set]
        pruned_documents.append(pruned_doc)
    return vocab, vocab_indices, pruned_documents