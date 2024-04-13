from llm.utiils import parse_json_string

json_string = """{
  "Название продукта": "йогурт «агуша» с персиком",
  "Категория продукта": "Молочные продукты",
  "Возрастная маркировка и рекомендованный возраст": "для питания детей старше 8 месяцев",
  "Наличие сахара": "5,8 г",
  "Рекламная информация": "",
  "Вес продукта в г.": "",
  "Состав продукта": [
    {
      "Ингредиенты": "молоко нормализованное, фруктовый наполнитель «персик» (сахар, вода, концентрированное персиковое пюре, крахмал кукурузный, загуститель — пектины, ароматизатор натуральный «персик», концентрированный сок из моркови (в качестве красящего вещества), концентрированный лимонный сок (для корректировки кислотности)}, пребиотик — олигофруктоза, концентрат сывороточных белков, закваска, пробиотические микроорганизмы — бифидобактерии (вв12)."
    }
  ],
  "Показатель пищевой ценности": {
    "Жиры": "2,7 r",
    "Белки": "2,8 г",
    "Углеводы": "9,4 r"
  },
  "Пребиотик": "0,6 г",
  "Кальций": "88 мг (14,7 %)",
  "Энергетическая ценность (калорийность)": {
    "В 100 г продукта": "307 кдж/?3 ккал"
  },
  "Содержание молочнокислых микроорганизмов в продукте не менее": "1x10° ee",
  "Процент от суточной нормы": "* — процент от суточной нормы."
}
"""

parsed_dict = parse_json_string(json_string)
print(parsed_dict['Показатель пищевой ценности'])
