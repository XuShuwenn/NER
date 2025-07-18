from transformers import BertForTokenClassification

class BertNER:
    """BERT模型类"""
    @staticmethod #这里用staticmethod是因为不需要访问类的任何属性或方法，直接从类中调用
    def from_pretrained(model_name, num_labels):
        model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        return model
        
