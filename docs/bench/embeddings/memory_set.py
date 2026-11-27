"""Доменный набор для memory-proxy: факты хранятся как в vec_facts («predicate: object»),
запросы — сообщения пользователя на разных языках. Для каждого запроса — множество релевантных фактов.
Цель — проверить именно то, что делает прокси: найти факт о пользователе по его сообщению."""
T = {  # id: (факт, [(язык, запрос), ...])
 1: ("works_at: Яндекс", [("ru","Напомни, в какой компании я сейчас работаю?"), ("ru2","Как обсудить повышение зарплаты с руководителем на моём нынешнем месте работы?"),
     ("en","Do you remember which company I work for?"), ("uk","Нагадай, у якій компанії я працюю?"), ("de","Weißt du noch, bei welcher Firma ich arbeite?")]),
 2: ("lives_in: Казань", [("ru","В каком городе я живу, ты помнишь?"), ("ru2","Куда сходить вечером в моём городе?"),
     ("en","Which city do I live in?"), ("uk","Яке місто я називав своїм домом?"), ("kk","Мен қай қалада тұрамын?")]),
 3: ("age: 34", [("ru","Сколько мне лет?"), ("ru2","Не поздно ли в моём возрасте начинать заниматься бегом?"),
     ("en","How old am I again?"), ("uk","Скільки мені років?"), ("de","Wie alt bin ich?")]),
 4: ("birthday: 14 марта", [("ru","Когда у меня день рождения?"), ("ru2","Помоги спланировать праздник на мой следующий ДР"),
     ("en","When is my birthday?"), ("uk","Коли в мене день народження?"), ("tr","Doğum günüm ne zaman?")]),
 5: ("partner_name: Анна", [("ru","Как зовут мою жену?"), ("ru2","Что подарить супруге на годовщину свадьбы?"),
     ("en","What's my wife's name?"), ("uk","Як звати мою дружину?"), ("mix","Хочу сделать surprise для my wife, есть идеи?")]),
 6: ("current_project: мобильное приложение для учёта расходов", [("ru","Над каким проектом я сейчас работаю?"), ("ru2","Какие библиотеки для графиков трат подойдут для моего приложения?"),
     ("en","Remind me what side project I'm building"), ("uk","Над яким проєктом я зараз працюю?"), ("de","Woran arbeite ich gerade für ein Projekt?")]),
 7: ("likes: зелёный чай", [("ru","Какой напиток я предпочитаю по утрам?"), ("ru2","Посоветуй, что заварить вечером, чтобы расслабиться"),
     ("en","What do I like to drink?"), ("uk","Що я люблю пити?"), ("de","Welches Getränk mag ich am liebsten?")]),
 8: ("dislikes: шумные вечеринки", [("ru","Какие мероприятия мне не нравятся?"), ("ru2","Друзья зовут на большую тусовку в клубе — стоит ли мне идти?"),
     ("en","Should I go to a loud club party this weekend, knowing me?"), ("uk","Чи варто мені йти на гучну вечірку?"), ("mix","Меня зовут на rave party, как думаешь, мне зайдёт?")]),
 9: ("knows_language: испанский", [("ru","Какие иностранные языки я знаю?"), ("ru2","Хочу поехать в Латинскую Америку — смогу ли я там объясниться?"),
     ("en","Which foreign languages do I speak?"), ("uk","Якими мовами я володію?"), ("es","¿Qué idiomas hablo?")]),
 10: ("uses_tool: Obsidian", [("ru","В каком приложении я веду заметки?"), ("ru2","Как мне лучше организовать базу знаний и связи между заметками?"),
     ("en","What note-taking app do I use?"), ("uk","У якій програмі я веду нотатки?"), ("de","Welche Notiz-App benutze ich?")]),
 11: ("has_skill: игра на гитаре", [("ru","На каком музыкальном инструменте я играю?"), ("ru2","Подбери песню, которую я смогу сыграть у костра"),
     ("en","What instrument can I play?"), ("uk","На якому інструменті я граю?"), ("tr","Hangi enstrümanı çalabiliyorum?")]),
 12: ("has_pet: кошка Муся", [("ru","Как зовут мою кошку?"), ("ru2","Чем лучше кормить мою кошку, она стала привередливой"),
     ("en","What's my cat's name?"), ("uk","Як звати мою кішку?"), ("kk","Менің мысығымның аты кім?")]),
 13: ("owns: Toyota RAV4", [("ru","Какая у меня машина?"), ("ru2","Когда пора менять резину на зимнюю на моём кроссовере?"),
     ("en","What car do I drive?"), ("uk","Яка в мене машина?"), ("de","Was für ein Auto fahre ich?")]),
 14: ("interested_in: астрономия", [("ru","Какое у меня хобби, связанное с наукой?"), ("ru2","Посоветуй телескоп для начинающего"),
     ("en","Recommend me a good book about stars and galaxies"), ("uk","Чим я цікавлюся в науці?"), ("mix","Хочу посмотреть на meteor shower, когда лучше?")]),
 15: ("likes: hiking in the mountains", [("ru","Чем я люблю заниматься на выходных на природе?"), ("ru2","Куда поехать в отпуск, чтобы походить по горам?"),
     ("en","What outdoor activity do I enjoy?"), ("uk","Що я люблю робити на природі?"), ("de","Was mache ich gern in der Natur?")]),
 16: ("uses_tool: Neovim", [("ru","Каким редактором кода я пользуюсь?"), ("ru2","Как настроить автодополнение для Python в моём редакторе?"),
     ("en","Which code editor do I use?"), ("uk","Яким редактором коду я користуюся?"), ("mix","Какие plugins поставить в мой editor?")]),
 17: ("has_skill: Kubernetes administration", [("ru","В чём я разбираюсь из DevOps?"), ("ru2","Как подготовиться к сертификации по оркестрации контейнеров?"),
     ("en","Am I experienced with container orchestration?"), ("uk","Що я вмію в DevOps?"), ("de","Kenne ich mich mit Container-Orchestrierung aus?")]),
 18: ("dislikes: кинза", [("ru","Какую зелень я не ем?"), ("ru2","Подбери рецепт гуакамоле с учётом моих вкусов"),
     ("en","Is there any herb I hate?"), ("uk","Яку зелень я не люблю?"), ("tr","Hangi otu sevmiyorum?")]),
 19: ("interested_in: история Древнего Рима", [("ru","Какая историческая эпоха мне интересна?"), ("ru2","Посоветуй сериал про гладиаторов и императоров"),
     ("en","Which period of history am I into?"), ("uk","Яка епоха історії мене цікавить?"), ("de","Welche Epoche der Geschichte interessiert mich?")]),
 20: ("owns: квартира в ипотеку", [("ru","Есть ли у меня кредит на жильё?"), ("ru2","Стоит ли мне досрочно гасить заём за квартиру?"),
     ("en","Do I have a mortgage?"), ("uk","Чи є в мене іпотека?"), ("kk","Менде ипотека бар ма?")]),
 21: ("knows_language: немецкий", [("ru","Я говорю по-немецки?"), ("ru2","Помоги подготовиться к экзамену Goethe-Zertifikat"),
     ("en","How good is my German?"), ("uk","Чи знаю я німецьку?"), ("de","Wie gut spreche ich Deutsch?")]),
 22: ("likes: фильмы Тарковского", [("ru","Какой режиссёр мне нравится?"), ("ru2","Посоветуй, что посмотреть из советского авторского кино"),
     ("en","Which film director do I like?"), ("uk","Який режисер мені подобається?"), ("de","Welchen Regisseur mag ich?")]),
 23: ("dislikes: ранние подъёмы", [("ru","Я жаворонок или сова?"), ("ru2","Как перестроить режим, если приходится вставать в шесть утра?"),
     ("en","Am I a morning person?"), ("uk","Я жайворонок чи сова?"), ("mix","Как мне привыкнуть к early wake up?")]),
 24: ("has_skill: выпечка хлеба на закваске", [("ru","Что я умею готовить?"), ("ru2","Почему мой домашний хлеб получается плотным?"),
     ("en","What do I bake at home?"), ("uk","Що я вмію пекти?"), ("de","Was backe ich zu Hause?")]),
 25: ("interested_in: инвестиции в облигации", [("ru","Во что я вкладываю деньги?"), ("ru2","Как ключевая ставка повлияет на мой портфель?"),
     ("en","What kind of investments am I into?"), ("uk","У що я інвестую?"), ("kk","Мен ақшамды неге саламын?")]),
 26: ("owns: велосипед Trek", [("ru","На чём я катаюсь по городу?"), ("ru2","Как подготовить мой байк к сезону?"),
     ("en","What bike do I have?"), ("uk","Який у мене велосипед?"), ("de","Was für ein Fahrrad habe ich?")]),
 27: ("uses_tool: PostgreSQL", [("ru","Какую СУБД я использую в проекте?"), ("ru2","Как ускорить медленные запросы в моей базе данных?"),
     ("en","Which database do I use?"), ("uk","Яку базу даних я використовую?"), ("mix","Как сделать index на jsonb в моей database?")]),
 28: ("likes: джаз", [("ru","Какую музыку я слушаю?"), ("ru2","Посоветуй концерт на выходные под мой вкус"),
     ("en","What music genre do I like?"), ("uk","Яку музику я слухаю?"), ("tr","Hangi müzik türünü severim?")]),
 29: ("dislikes: долгие созвоны", [("ru","Что меня раздражает в работе?"), ("ru2","Как сократить количество встреч в календаре?"),
     ("en","What annoys me at work?"), ("uk","Що мене дратує на роботі?"), ("de","Was nervt mich bei der Arbeit?")]),
 30: ("interested_in: шахматы", [("ru","Какая настольная игра мне интересна?"), ("ru2","Как улучшить игру в эндшпиле?"),
     ("en","Which board game am I into?"), ("uk","Яка настільна гра мене цікавить?"), ("kk","Маған қандай үстел ойыны қызық?")]),
}
D = [  # дистракторы: правдоподобные факты того же пользователя, не отвечающие на запросы
 "likes: итальянская кухня", "likes: осень", "likes: настольный теннис", "likes: романы Пелевина", "likes: подкасты про экономику",
 "likes: море", "likes: фотография на плёнку", "likes: stand-up comedy", "likes: indie games", "likes: тёплые свитера",
 "dislikes: метро в час пик", "dislikes: спам-звонки", "dislikes: грибы", "dislikes: фильмы ужасов", "dislikes: бюрократия",
 "dislikes: жару", "dislikes: очереди",
 "knows_language: английский", "knows_language: татарский",
 "uses_tool: Telegram", "uses_tool: Excel", "uses_tool: Figma", "uses_tool: Git", "uses_tool: Jira", "uses_tool: Google Calendar",
 "has_skill: вождение", "has_skill: плавание", "has_skill: публичные выступления", "has_skill: вязание", "has_skill: ремонт компьютеров",
 "has_skill: touch typing", "has_skill: первая помощь",
 "owns: MacBook Pro", "owns: Kindle Paperwhite", "owns: PlayStation 5", "owns: робот-пылесос", "owns: кофемашина",
 "interested_in: психология", "interested_in: архитектура модерна", "interested_in: машинное обучение", "interested_in: йога",
 "interested_in: путешествия по Азии", "interested_in: садоводство", "interested_in: Формула-1", "interested_in: японская культура",
]
# запросы, на которые законно отвечают и дистракторы
EXTRA = {(9, "ru"): ["knows_language: английский", "knows_language: татарский"], (9, "en"): ["knows_language: английский", "knows_language: татарский"],
         (9, "uk"): ["knows_language: английский", "knows_language: татарский"], (9, "es"): ["knows_language: английский", "knows_language: татарский"],
         (24, "ru"): [], (29, "ru"): ["dislikes: бюрократия"], (29, "en"): ["dislikes: бюрократия"], (29, "uk"): ["dislikes: бюрократия"], (29, "de"): ["dislikes: бюрократия"]}

def build():
    facts = [T[i][0] for i in sorted(T)] + D
    idx = {f: n for n, f in enumerate(facts)}
    queries = []
    for i in sorted(T):
        for lang, q in T[i][1]:
            rel = {idx[T[i][0]]} | {idx[f] for f in EXTRA.get((i, lang), [])}
            if (i in (9, 21)) and lang in ("ru", "en", "uk", "es") and i == 9: rel.add(idx[T[21][0]])
            queries.append((lang, q, rel, i))
    return facts, queries
