import re
import string

class Preprocessing():

    # Input: string
    # Output: processed_string

    def __init__(self, stopwords_path, annotator_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            self.stopwords = [line.strip() for line in f.readlines()]

        from vncorenlp import VnCoreNLP
        self.annotator = VnCoreNLP(annotator_path, annotators="wseg,pos,ner,parse", max_heap_size='-Xmx2g')

    @staticmethod
    def clean_text(text):
        text = text.lower()
        text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
        return text

    def stop_word_remove(self, text):
        words = re.findall(r'\w+', text)
        filtered_words = [word for word in words if word not in self.stopwords or word == "không"]
        filtered_text = ' '.join(filtered_words)
        return filtered_text

    @staticmethod
    def remove_number( text):
        return re.sub(r'\d+', ' ', text)

    @staticmethod
    def remove_special_char(string):
        # Removing characters that are neither alphanumeric nor basic punctuations,
        # but preserving Vietnamese diacritics.
        cleaned_string = re.sub(
            r"[^a-zA-Z0-9.,!? áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđĐ ]", '', string)
        return cleaned_string

    @staticmethod
    def replace_abbreviations(text):
        # Creating a dictionary to store abbreviation mappings
        abbreviations = {
            r'\bks\b': 'khách sạn',
            # The \b ensures we match whole words only, avoiding replacements in the middle of words
            r'\bnv\b': 'nhân viên',

            r'\bko\b': 'không',

            r'\bdc\b': 'được',

            r'\bđc\b': 'được',

            r'\btk\b': 'tài khoản',

            r'\bh\b': 'giờ'
        }

        # Looping through the dictionary to replace all abbreviations
        for abbr, full_form in abbreviations.items():
            text = re.sub(abbr, full_form, text, flags=re.IGNORECASE)

        return text

    def segment(self, text):
        annotated_text = self.annotator.annotate(text)
        segmented_sentence = " ".join([word['form'] for sentence in annotated_text['sentences'] for word in sentence])
        return segmented_sentence

    def process(self, text):
        text = self.clean_text(text)
        text = self.remove_number(text)
        text = self.remove_special_char(text)
        text = self.replace_abbreviations(text)
        text = self.segment(text)
        text = self.stop_word_remove(text)
        return text

if __name__ == '__main__':
    preprocessor = Preprocessing('../data/external/vietnamese-stopwords-dash.txt',
                                 './libs/vncorenlp/VnCoreNLP-1.2.jar')
    text = 'Điểm mua sắm quà lưu niệm lý tưởng, đầy đủ hàng hóa của các làng nghề truyền thống của vùng quê đất võ Tây Sơn, Bình Định. Nơi đây là điểm cung ứng hàng hóa của người nông dân nên giá cũng rất bình dân. Hãy đến đây khi đến với Bình Định, vì không muốn khi về quên mang theo món đặc sản truyền thống, tinh hoa của đất võ Tây Sơn 😍'
    text = preprocessor.process(text)
    print(text)