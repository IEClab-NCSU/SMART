from keyphrase_extractor_v2 import *
import shutil
import os


def create_and_save_df(docs, targets, preds, course, algorithm):
    df = pd.DataFrame()
    df['Original'] =docs
    df['Target'] = targets
    df['Predicted'] = preds

    df.to_pickle(f'AKE_Result_Dataframes/{course}_{algorithm}_df.pkl')

def main():
    # create a folder to save result dataframes
    dataframe_path = 'AKE_Result_Dataframes/'

    if os.path.exists(dataframe_path):
        shutil.rmtree(dataframe_path)

    os.mkdir(dataframe_path)

    # generate and save AKE results for OLI Introduction to Biology and General Chemistry 1 courses
    # for all baseline AKE methods.
    for course in ['oli-intro-bio', 'oli-gen-chem']:
        keyphrase_extractor = KeyphraseExtractor(course, verbose=True)
        keyphrase_extractor.run_keybert(1, verbose=True)
        keyphrase_extractor.evaluate(verbose=True)
        create_and_save_df(keyphrase_extractor.original_documents_whole, keyphrase_extractor.document_labels,
                           keyphrase_extractor.final_keyphrases, course, 'keybert')

        keyphrase_extractor.reinitialize()
        keyphrase_extractor.run_TextRank(1, verbose=True)
        keyphrase_extractor.evaluate(verbose=True)
        create_and_save_df(keyphrase_extractor.original_documents_whole, keyphrase_extractor.document_labels,
                           keyphrase_extractor.final_keyphrases, course, 'textrank')

        keyphrase_extractor.reinitialize()
        keyphrase_extractor.run_SingleRank(1, verbose=True)
        keyphrase_extractor.evaluate(verbose=True)
        create_and_save_df(keyphrase_extractor.original_documents_whole, keyphrase_extractor.document_labels,
                           keyphrase_extractor.final_keyphrases, course, 'singlerank')

        keyphrase_extractor.reinitialize()
        keyphrase_extractor.run_TopicRank(1, verbose=True)
        keyphrase_extractor.evaluate(verbose=True)
        create_and_save_df(keyphrase_extractor.original_documents_whole, keyphrase_extractor.document_labels,
                           keyphrase_extractor.final_keyphrases, course, 'topicrank')

        keyphrase_extractor.reinitialize()
        keyphrase_extractor.run_MultipartiteRank(1, verbose=True)
        keyphrase_extractor.evaluate(verbose=True)
        create_and_save_df(keyphrase_extractor.original_documents_whole, keyphrase_extractor.document_labels,
                           keyphrase_extractor.final_keyphrases, course, 'multipartiterank')

        keyphrase_extractor.reinitialize()
        keyphrase_extractor.run_RAKE(1, verbose=True)
        keyphrase_extractor.evaluate(verbose=True)
        create_and_save_df(keyphrase_extractor.original_documents_whole, keyphrase_extractor.document_labels,
                           keyphrase_extractor.final_keyphrases, course, 'rake')

        keyphrase_extractor.reinitialize()
        keyphrase_extractor.run_YAKE(1, verbose=True)
        keyphrase_extractor.evaluate(verbose=True)
        create_and_save_df(keyphrase_extractor.original_documents_whole, keyphrase_extractor.document_labels,
                           keyphrase_extractor.final_keyphrases, course, 'yake')


if __name__ == '__main__':
    main()
