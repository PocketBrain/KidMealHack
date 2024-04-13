SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты помогаешь людям проверить качество товара на основе текста состава."
#SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты помогаешь людям проверить качество товара на основе текста состава. Пожалуйста, возвращай ответы только в честном формате, без добавления лишней информации."
FROM_TEXT_2_JSON_PROMPT = """ "Highlight important data on the composition and nutritional value from the text about baby food and format them in a convenient json format for subsequent analysis. Please note that this data includes information about the composition of the product, its nutritional value, as well as storage and preparation conditions"
    Я даю на вход текст состава продукта, твоя задача выделить из текста слова и сформировать результат в формате "Параметр": "Значение". 
    Текст имеет ошибки в словах, ты должен их исправить и представить данные в формате JSON, исправляй слова, если они не правильно написаны.
    Выделяй следующие параметры, формат JSON:
    
    {
    "name" : Название продукта. String.
    "brand" : название бренда продукта. String.
    "category" : Категория продукта, выбирай только из списка: (Cухие каши и крахмалистые продукты, Молочные продукты, Фруктовые и овощные пюре/коктейли и фруктовые десерты,  Сухие закуски и перекусы, Ингредиенты, Кондитерские изделия, Напитки). Category.
    "old" : Возрастная маркировка и рекомендованный возраст. String.
    "HasSugar" : Наличие сахара. Boolean.
    "HasSodium" : Наличие натрия. Boolean.
    "HasSubSugar" : Наличие подсластителей. Boolean.
    "HasTransFat" : Наличие транс-жиров. Boolean.
    "HasGMO" : Наличие ГМО. Boolean.
    "kcal": - Количество килокалорий в продукте. Integer.
    "composition" : Состав продукта через запятую. String.
    "proteins" : Показатель белков и протеина. Float.
    "fats" : Показатель жиров. Float.
    "carbohydrates" : Показатель углеводов. Float.
    "energy" : Энергетическая ценность. Float.
    "HasMarketingLabels" : Рекламные заголовки в тексте. Boolean.
    }
    
    Текст описания товара:
    
    """

FROM_JSON_2_RULE_PROMPT = """Теперь твоя задача понять на основе данного JSON определить выполняется ли требования к товару:
Категория: Cухие каши и крахмалистые продукты. Энергетическая ценность (ккал/100 г) >= 80. Натрий (мг/ 100 ккал) <= 50. Добавленные свободные сахара или подсластитель – нет. Общий белок (г/100ккал) и вес белка <= 5,5 г (если содержит молоко). Общее количество жиров (г/100ккал) (без транс-жиров) <= 4,5г. Содержание фруктов(% веса) <= 10% сухого веса. Возрастная маркировка (месяцы) - 6-36
Категория: Молочные продукты. Энергетическая ценность (ккал/100 г) >= 60. Натрий (мг/ 100 ккал) <= 50. Добавленные свободные сахара или подсластитель – нет. Общее количество жиров (г/100ккал) (без транс-жиров) <= 4,5г. Содержание фруктов(% веса) <= 5%. Возрастная маркировка (месяцы) - 6-36
Категория: Фруктовые и овощные пюре/коктейли и фруктовые десерты. Энергетическая ценность (ккал/100 г) >= 60. Натрий (мг/ 100 ккал) <= 50. Добавленные свободные сахара или подсластитель – нет. Общий белок (г/100ккал) и вес белка <= 5,5 г (если содержит молоко). Общее количество жиров (г/100ккал) (без транс-жиров) <= 4,5г. Содержание фруктов(% веса) <= 10% сухого веса. Возрастная маркировка (месяцы) - 6-36
Категория: Сухие закуски и перекусы. Энергетическая ценность (ккал/100 г) <= 50. Натрий (мг/ 100 ккал) <= 50. Общий сахар <= 15%. Добавленные свободные сахара или подсластитель – нет. Общий белок (г/100ккал) и вес белка <= 5,5 г (если содержит молоко). Общее количество жиров (г/100ккал) (без транс-жиров) <= 4,5г. Содержание фруктов(% веса) – 100%. Возрастная маркировка (месяцы) - 6-36
Если категория товара Кондитерские изделия или Напитки, то такой продукт нельзя употреблять детям до 3 лет, и ты должен сказать, это плохой товар по оценке ВОЗ.
Ни один продукт не может содержать промышленно произведенные транс-жирные кислоты.
Продукты, приготовленные из смеси или пюре, должны иметь верхний возрастной предел 12 месяцев.
На упаковках или в сопутствующих маркетинговых материалах не допускаются никакие заявления о полезном составе, диетологических доводах о пользе продукта для здоровья или рекламные тезисы.
Продукт не должен иметь рекламные заголовки в тексте. если они имеются напиши какие.
Если состав продукта и его описание соответствует требованиям, то напиши что продукта можно использовать, и подробно распиши почему на основе требований к товару.
Если продукт не соответствует требованиям, то напиши что его нельзя использовать и подробно распиши почему на основе требований к товару.
"""

MODEL_PATH = "model-q8_0.gguf"

SYSTEM_TOKEN = 1587
USER_TOKEN = 2188
BOT_TOKEN = 12435
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}