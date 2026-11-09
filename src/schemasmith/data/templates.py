"""Compositional support-ticket pieces in eight languages.

A ticket is a header, optional chatter, a scenario body, a severity cue, a
tier mention, and optional order and refund mentions. Ticket ids, order
references, and amounts are inserted by the generator. Category names are not
spelled out in the bodies.

Held-out design: one sub-scenario per category, one header format group, the
third severity phrasing, and the third opener and closer are used only for the
test split, so no test ticket shares its body, header, or severity sentence
with a training ticket.
"""

from __future__ import annotations

SCENARIOS: tuple[str, ...] = ("billing", "bug", "access", "feature_request")

SUBSCENARIOS: dict[str, tuple[str, ...]] = {
    "billing": (
        "double_charge",
        "cancelled_still_charged",
        "wrong_amount",
        "refund_status",
        "invoice_error",
    ),
    "bug": (
        "crash_on_save",
        "sync_slow",
        "export_broken",
        "images_missing",
        "duplicate_notifications",
    ),
    "access": (
        "password_reset",
        "sso_error",
        "two_factor_lost",
        "account_locked",
        "invite_expired",
    ),
    "feature_request": ("csv_export", "bulk_edit", "dark_mode", "api_webhooks", "calendar_sync"),
}

# Index 1 of each category is held out for the test split.
TEST_SUBSCENARIOS: frozenset[str] = frozenset(subs[1] for subs in SUBSCENARIOS.values())

# Billing sub-scenarios that carry a refund request.
REFUND_SUBSCENARIOS: frozenset[str] = frozenset(
    {"double_charge", "cancelled_still_charged", "wrong_amount", "refund_status"}
)

# Header formats. Train uses 0-2, test uses 3-4.
TRAIN_HEADERS: tuple[int, ...] = (0, 1, 2)
TEST_HEADERS: tuple[int, ...] = (3, 4)

# Severity phrasings: 0 and 1 for train, 2 for test.
TRAIN_CUES: tuple[int, ...] = (0, 1)
TEST_CUES: tuple[int, ...] = (2,)

L: dict[str, dict] = {}

L["en"] = {
    "from": "From",
    "subject": "Subject",
    "subjects": {
        "billing": "problem with my bill",
        "bug": "something is broken",
        "access": "can't get in",
        "feature_request": "suggestion",
    },
    "body": {
        "double_charge": (
            "I was charged twice for the same month.",
            "My card shows two identical payments this cycle.",
        ),
        "cancelled_still_charged": (
            "I cancelled my subscription last month but you still took a payment.",
            "Even after cancelling, another charge hit my card.",
        ),
        "wrong_amount": (
            "The amount you billed is higher than the price on the website.",
            "I was billed more than my plan costs.",
        ),
        "refund_status": (
            "I was promised a refund weeks ago and it still has not arrived.",
            "Support approved my refund but I never received it.",
        ),
        "invoice_error": (
            "The company name and tax ID on my invoice are wrong.",
            "My invoice lists the wrong billing address.",
        ),
        "crash_on_save": (
            "The editor freezes and closes every time I save a draft.",
            "Saving a document makes the app crash.",
        ),
        "sync_slow": (
            "Changes take ages to show up on my other devices.",
            "Syncing between laptop and phone has become extremely slow.",
        ),
        "export_broken": (
            "Exporting a report to PDF gives me an empty file.",
            "The export button spins forever and nothing downloads.",
        ),
        "images_missing": (
            "Images I upload show up as broken icons.",
            "Pictures disappear from my pages after I publish them.",
        ),
        "duplicate_notifications": (
            "I get every notification three or four times.",
            "The app keeps sending me the same alert over and over.",
        ),
        "password_reset": (
            "The password reset email never reaches my inbox.",
            "I asked for a new password link but nothing arrives.",
        ),
        "sso_error": (
            "Single sign-on with our company login throws an error page.",
            "When I sign in through our identity provider I get bounced back to the login screen.",
        ),
        "two_factor_lost": (
            "I lost the phone with my authenticator app and cannot pass two-factor.",
            "My two-factor codes stopped working after I changed phones.",
        ),
        "account_locked": (
            "My account was locked after a few wrong passwords.",
            "The site says my account is locked and I cannot sign in.",
        ),
        "invite_expired": (
            "The invitation link from my colleague says it has expired.",
            "I cannot join my team because the invite link no longer works.",
        ),
        "csv_export": (
            "It would help a lot if we could download our data as CSV.",
            "Could you add a spreadsheet export option?",
        ),
        "bulk_edit": (
            "Please let us change many records at once instead of one by one.",
            "We would love a way to edit a whole batch of items together.",
        ),
        "dark_mode": (
            "A dark theme would make the app easier on the eyes.",
            "Please consider adding a night mode.",
        ),
        "api_webhooks": (
            "We would like webhooks so our systems hear about changes right away.",
            "Could the API push events to our server?",
        ),
        "calendar_sync": (
            "It would be great to sync deadlines with my calendar.",
            "Please add an option to show due dates in Google or Outlook calendar.",
        ),
    },
    "cue": {
        4: (
            "Our whole team is blocked and production is down.",
            "This is an outage for everyone at our company.",
            "Nothing works for any of our users right now, it is an emergency.",
        ),
        3: (
            "This blocks my own work today, though my colleagues are fine.",
            "I cannot get my job done until this is fixed.",
            "I am stuck and cannot continue with my tasks.",
        ),
        2: (
            "It is annoying, but I have a workaround for now.",
            "It slows me down, but there is a workaround.",
            "There is a clumsy workaround, so it can wait a little.",
        ),
        1: (
            "No rush at all, nothing is blocked.",
            "Just a low-priority note, take your time.",
            "Purely a nice-to-have whenever you get to it.",
        ),
    },
    "tier": ("I am on the {tier} plan.", "(account type: {tier})"),
    "order": ("Order: {order}.", "This is about order {order}."),
    "refund": ("Please refund {amount} USD.", "I would like {amount} USD back."),
    "open": ("Hi team,", "Hello,", "Good morning support,"),
    "close": ("Thanks.", "Best regards.", "Cheers and thanks in advance."),
}

L["es"] = {
    "from": "De",
    "subject": "Asunto",
    "subjects": {
        "billing": "problema con mi factura",
        "bug": "algo no funciona",
        "access": "no puedo entrar",
        "feature_request": "sugerencia",
    },
    "body": {
        "double_charge": (
            "Me cobraron dos veces el mismo mes.",
            "En mi tarjeta aparecen dos pagos iguales este ciclo.",
        ),
        "cancelled_still_charged": (
            "Cancelé mi suscripción el mes pasado y aun así me cobraron.",
            "Después de cancelar, llegó otro cargo a mi tarjeta.",
        ),
        "wrong_amount": (
            "El importe cobrado es mayor que el precio de la web.",
            "Me facturaron más de lo que cuesta mi plan.",
        ),
        "refund_status": (
            "Me prometieron un reembolso hace semanas y todavía no llega.",
            "Soporte aprobó mi reembolso pero nunca lo recibí.",
        ),
        "invoice_error": (
            "El nombre de la empresa y el NIF de mi factura están mal.",
            "Mi factura tiene una dirección de facturación equivocada.",
        ),
        "crash_on_save": (
            "El editor se congela y se cierra cada vez que guardo un borrador.",
            "Al guardar un documento la aplicación se cae.",
        ),
        "sync_slow": (
            "Los cambios tardan muchísimo en aparecer en mis otros dispositivos.",
            "La sincronización entre portátil y móvil se ha vuelto lentísima.",
        ),
        "export_broken": (
            "Al exportar un informe a PDF obtengo un archivo vacío.",
            "El botón de exportar gira sin parar y no se descarga nada.",
        ),
        "images_missing": (
            "Las imágenes que subo aparecen como iconos rotos.",
            "Las fotos desaparecen de mis páginas después de publicarlas.",
        ),
        "duplicate_notifications": (
            "Recibo cada notificación tres o cuatro veces.",
            "La aplicación me manda la misma alerta una y otra vez.",
        ),
        "password_reset": (
            "El correo para restablecer la contraseña nunca llega.",
            "Pedí un enlace de contraseña nueva pero no recibo nada.",
        ),
        "sso_error": (
            "El inicio de sesión único con la cuenta de la empresa muestra una página de error.",
            "Al entrar con nuestro proveedor de identidad vuelvo a la pantalla de acceso.",
        ),
        "two_factor_lost": (
            "Perdí el teléfono con la app de autenticación y no paso la verificación en dos pasos.",
            "Mis códigos de doble factor dejaron de funcionar al cambiar de móvil.",
        ),
        "account_locked": (
            "Mi cuenta se bloqueó después de varias contraseñas incorrectas.",
            "La web dice que mi cuenta está bloqueada y no puedo entrar.",
        ),
        "invite_expired": (
            "El enlace de invitación de mi compañero dice que caducó.",
            "No puedo unirme a mi equipo porque la invitación ya no funciona.",
        ),
        "csv_export": (
            "Nos ayudaría mucho poder descargar los datos en CSV.",
            "¿Podrían añadir una opción para exportar a hoja de cálculo?",
        ),
        "bulk_edit": (
            "Por favor permitan cambiar muchos registros a la vez y no uno por uno.",
            "Nos encantaría poder editar un lote entero de elementos juntos.",
        ),
        "dark_mode": (
            "Un tema oscuro haría la aplicación más cómoda para la vista.",
            "Por favor consideren añadir un modo nocturno.",
        ),
        "api_webhooks": (
            "Queremos webhooks para que nuestros sistemas se enteren de los cambios al instante.",
            "¿La API podría enviar eventos a nuestro servidor?",
        ),
        "calendar_sync": (
            "Sería genial sincronizar las fechas límite con mi calendario.",
            "Añadan una opción para ver los vencimientos en Google o Outlook.",
        ),
    },
    "cue": {
        4: (
            "Todo nuestro equipo está bloqueado y producción está caída.",
            "Es una caída para toda la empresa.",
            "Ahora mismo no funciona nada para ningún usuario, es una emergencia.",
        ),
        3: (
            "Esto bloquea mi trabajo de hoy, aunque mis compañeros están bien.",
            "No puedo hacer mi trabajo hasta que se arregle.",
            "Estoy atascado y no puedo seguir con mis tareas.",
        ),
        2: (
            "Es molesto, pero por ahora tengo una solución alternativa.",
            "Me retrasa, pero hay una forma de evitarlo.",
            "Hay un apaño torpe, así que puede esperar un poco.",
        ),
        1: (
            "Sin ninguna prisa, nada está bloqueado.",
            "Solo una nota de baja prioridad, con calma.",
            "Es algo deseable para cuando puedan.",
        ),
    },
    "tier": ("Tengo el plan {tier}.", "(tipo de cuenta: {tier})"),
    "order": ("Pedido: {order}.", "Se trata del pedido {order}."),
    "refund": ("Por favor reembolsen {amount} USD.", "Quiero que me devuelvan {amount} USD."),
    "open": ("Hola equipo,", "Buenas,", "Buenos días, soporte:"),
    "close": ("Gracias.", "Saludos.", "Un saludo y gracias de antemano."),
}

L["fr"] = {
    "from": "De",
    "subject": "Objet",
    "subjects": {
        "billing": "souci de facturation",
        "bug": "quelque chose ne marche pas",
        "access": "impossible de me connecter",
        "feature_request": "suggestion",
    },
    "body": {
        "double_charge": (
            "J'ai été débité deux fois pour le même mois.",
            "Ma carte affiche deux paiements identiques ce mois-ci.",
        ),
        "cancelled_still_charged": (
            "J'ai résilié mon abonnement le mois dernier mais vous avez encore prélevé.",
            "Même après la résiliation, un nouveau prélèvement est passé.",
        ),
        "wrong_amount": (
            "Le montant facturé est plus élevé que le prix affiché sur le site.",
            "On m'a facturé plus que le prix de mon offre.",
        ),
        "refund_status": (
            "On m'a promis un remboursement il y a des semaines et rien n'est arrivé.",
            "Le support a validé mon remboursement mais je ne l'ai jamais reçu.",
        ),
        "invoice_error": (
            "Le nom de la société et le numéro de TVA sur ma facture sont faux.",
            "Ma facture indique une mauvaise adresse de facturation.",
        ),
        "crash_on_save": (
            "L'éditeur se fige et se ferme chaque fois que j'enregistre un brouillon.",
            "Enregistrer un document fait planter l'application.",
        ),
        "sync_slow": (
            "Les modifications mettent une éternité à apparaître sur mes autres appareils.",
            "La synchronisation entre ordinateur et téléphone est devenue très lente.",
        ),
        "export_broken": (
            "L'export d'un rapport en PDF donne un fichier vide.",
            "Le bouton d'export tourne sans fin et rien ne se télécharge.",
        ),
        "images_missing": (
            "Les images que j'envoie s'affichent comme des icônes cassées.",
            "Les photos disparaissent de mes pages après publication.",
        ),
        "duplicate_notifications": (
            "Je reçois chaque notification trois ou quatre fois.",
            "L'application m'envoie la même alerte encore et encore.",
        ),
        "password_reset": (
            "L'e-mail de réinitialisation du mot de passe n'arrive jamais.",
            "J'ai demandé un lien de nouveau mot de passe mais je ne reçois rien.",
        ),
        "sso_error": (
            "La connexion unique avec le compte de l'entreprise affiche une page d'erreur.",
            "Quand je passe par notre fournisseur d'identité, je reviens à l'écran de connexion.",
        ),
        "two_factor_lost": (
            "J'ai perdu le téléphone avec l'appli d'authentification et je ne passe plus la double authentification.",
            "Mes codes de double authentification ne marchent plus depuis que j'ai changé de téléphone.",
        ),
        "account_locked": (
            "Mon compte a été verrouillé après quelques mauvais mots de passe.",
            "Le site dit que mon compte est verrouillé et je ne peux pas entrer.",
        ),
        "invite_expired": (
            "Le lien d'invitation de mon collègue indique qu'il a expiré.",
            "Je ne peux pas rejoindre mon équipe car l'invitation ne marche plus.",
        ),
        "csv_export": (
            "Ce serait très utile de pouvoir télécharger nos données en CSV.",
            "Pourriez-vous ajouter un export vers tableur ?",
        ),
        "bulk_edit": (
            "Merci de permettre la modification de nombreux éléments en une fois plutôt qu'un par un.",
            "Nous aimerions pouvoir éditer tout un lot d'éléments ensemble.",
        ),
        "dark_mode": (
            "Un thème sombre reposerait les yeux.",
            "Pensez à ajouter un mode nuit, s'il vous plaît.",
        ),
        "api_webhooks": (
            "Nous voudrions des webhooks pour que nos systèmes soient prévenus tout de suite.",
            "L'API pourrait-elle envoyer des événements à notre serveur ?",
        ),
        "calendar_sync": (
            "Ce serait super de synchroniser les échéances avec mon agenda.",
            "Ajoutez une option pour voir les dates limites dans Google ou Outlook.",
        ),
    },
    "cue": {
        4: (
            "Toute notre équipe est bloquée et la production est à l'arrêt.",
            "C'est une panne pour toute l'entreprise.",
            "Plus rien ne marche pour aucun utilisateur, c'est une urgence.",
        ),
        3: (
            "Cela bloque mon travail aujourd'hui, mes collègues n'ont pas le problème.",
            "Je ne peux pas faire mon travail tant que ce n'est pas réglé.",
            "Je suis coincé et je ne peux pas continuer mes tâches.",
        ),
        2: (
            "C'est pénible, mais j'ai une solution de contournement pour l'instant.",
            "Ça me ralentit, mais il existe un contournement.",
            "Il y a un contournement bancal, donc ça peut attendre un peu.",
        ),
        1: (
            "Aucune urgence, rien n'est bloqué.",
            "Simple remarque peu prioritaire, prenez votre temps.",
            "C'est juste un plus, quand vous aurez le temps.",
        ),
    },
    "tier": ("J'ai l'offre {tier}.", "(type de compte : {tier})"),
    "order": ("Commande : {order}.", "Il s'agit de la commande {order}."),
    "refund": ("Merci de rembourser {amount} USD.", "Je souhaite récupérer {amount} USD."),
    "open": ("Bonjour l'équipe,", "Bonjour,", "Bonjour le support,"),
    "close": ("Merci.", "Cordialement.", "Merci d'avance et bonne journée."),
}

L["de"] = {
    "from": "Von",
    "subject": "Betreff",
    "subjects": {
        "billing": "Problem mit der Rechnung",
        "bug": "etwas funktioniert nicht",
        "access": "komme nicht rein",
        "feature_request": "Vorschlag",
    },
    "body": {
        "double_charge": (
            "Mir wurde derselbe Monat zweimal berechnet.",
            "Auf meiner Karte stehen diesen Monat zwei gleiche Zahlungen.",
        ),
        "cancelled_still_charged": (
            "Ich habe mein Abo letzten Monat gekündigt, trotzdem wurde abgebucht.",
            "Auch nach der Kündigung kam eine weitere Abbuchung.",
        ),
        "wrong_amount": (
            "Der berechnete Betrag ist höher als der Preis auf der Website.",
            "Mir wurde mehr berechnet, als mein Tarif kostet.",
        ),
        "refund_status": (
            "Mir wurde vor Wochen eine Erstattung zugesagt, sie ist immer noch nicht da.",
            "Der Support hat meine Erstattung genehmigt, angekommen ist sie nie.",
        ),
        "invoice_error": (
            "Firmenname und Steuernummer auf meiner Rechnung sind falsch.",
            "Auf meiner Rechnung steht die falsche Rechnungsadresse.",
        ),
        "crash_on_save": (
            "Der Editor friert ein und schließt sich bei jedem Speichern eines Entwurfs.",
            "Beim Speichern eines Dokuments stürzt die App ab.",
        ),
        "sync_slow": (
            "Änderungen brauchen ewig, bis sie auf meinen anderen Geräten erscheinen.",
            "Die Synchronisierung zwischen Laptop und Handy ist extrem langsam geworden.",
        ),
        "export_broken": (
            "Beim Export eines Berichts als PDF bekomme ich eine leere Datei.",
            "Der Export-Knopf dreht sich endlos und nichts wird heruntergeladen.",
        ),
        "images_missing": (
            "Hochgeladene Bilder erscheinen als kaputte Symbole.",
            "Fotos verschwinden nach dem Veröffentlichen von meinen Seiten.",
        ),
        "duplicate_notifications": (
            "Ich bekomme jede Benachrichtigung drei- oder viermal.",
            "Die App schickt mir immer wieder dieselbe Meldung.",
        ),
        "password_reset": (
            "Die E-Mail zum Zurücksetzen des Passworts kommt nie an.",
            "Ich habe einen Link für ein neues Passwort angefordert, aber nichts kommt.",
        ),
        "sso_error": (
            "Die Einmalanmeldung mit dem Firmenkonto zeigt eine Fehlerseite.",
            "Über unseren Identitätsanbieter lande ich immer wieder auf der Anmeldeseite.",
        ),
        "two_factor_lost": (
            "Ich habe das Handy mit der Authenticator-App verloren und komme nicht durch die Zwei-Faktor-Prüfung.",
            "Meine Zwei-Faktor-Codes funktionieren seit dem Handywechsel nicht mehr.",
        ),
        "account_locked": (
            "Mein Konto wurde nach ein paar falschen Passwörtern gesperrt.",
            "Die Seite sagt, mein Konto ist gesperrt, und ich kann mich nicht anmelden.",
        ),
        "invite_expired": (
            "Der Einladungslink meiner Kollegin sagt, er sei abgelaufen.",
            "Ich kann meinem Team nicht beitreten, weil die Einladung nicht mehr geht.",
        ),
        "csv_export": (
            "Es würde sehr helfen, unsere Daten als CSV herunterzuladen.",
            "Könnt ihr einen Export als Tabelle einbauen?",
        ),
        "bulk_edit": (
            "Bitte lasst uns viele Einträge auf einmal ändern statt einzeln.",
            "Wir hätten gern eine Möglichkeit, einen ganzen Stapel Einträge gemeinsam zu bearbeiten.",
        ),
        "dark_mode": (
            "Ein dunkles Design wäre angenehmer für die Augen.",
            "Bitte überlegt euch einen Nachtmodus.",
        ),
        "api_webhooks": (
            "Wir möchten Webhooks, damit unsere Systeme Änderungen sofort mitbekommen.",
            "Könnte die API Ereignisse an unseren Server schicken?",
        ),
        "calendar_sync": (
            "Es wäre toll, Fristen mit meinem Kalender zu synchronisieren.",
            "Bitte eine Option, Fälligkeiten in Google oder Outlook anzuzeigen.",
        ),
    },
    "cue": {
        4: (
            "Unser ganzes Team ist blockiert und die Produktion steht.",
            "Das ist ein Ausfall für die ganze Firma.",
            "Gerade funktioniert für keinen Nutzer irgendetwas, das ist ein Notfall.",
        ),
        3: (
            "Das blockiert heute meine Arbeit, meine Kollegen sind nicht betroffen.",
            "Ich kann meine Arbeit nicht machen, bis das behoben ist.",
            "Ich hänge fest und komme mit meinen Aufgaben nicht weiter.",
        ),
        2: (
            "Es nervt, aber ich habe vorerst einen Workaround.",
            "Es bremst mich, aber es gibt eine Umgehung.",
            "Es gibt einen umständlichen Umweg, also kann es etwas warten.",
        ),
        1: (
            "Keine Eile, nichts ist blockiert.",
            "Nur ein Hinweis mit niedriger Priorität, lasst euch Zeit.",
            "Einfach nett zu haben, wann immer ihr dazu kommt.",
        ),
    },
    "tier": ("Ich habe den Tarif {tier}.", "(Kontotyp: {tier})"),
    "order": ("Bestellung: {order}.", "Es geht um die Bestellung {order}."),
    "refund": ("Bitte erstatten Sie {amount} USD.", "Ich hätte gern {amount} USD zurück."),
    "open": ("Hallo Team,", "Guten Tag,", "Moin Support,"),
    "close": ("Danke.", "Viele Grüße.", "Danke im Voraus und schöne Grüße."),
}

L["pt"] = {
    "from": "De",
    "subject": "Assunto",
    "subjects": {
        "billing": "problema na cobrança",
        "bug": "algo não funciona",
        "access": "não consigo entrar",
        "feature_request": "sugestão",
    },
    "body": {
        "double_charge": (
            "Fui cobrado duas vezes pelo mesmo mês.",
            "Meu cartão mostra dois pagamentos iguais neste ciclo.",
        ),
        "cancelled_still_charged": (
            "Cancelei minha assinatura no mês passado e mesmo assim cobraram.",
            "Mesmo depois de cancelar, caiu outra cobrança no cartão.",
        ),
        "wrong_amount": (
            "O valor cobrado é maior que o preço do site.",
            "Fui cobrado mais do que custa o meu plano.",
        ),
        "refund_status": (
            "Prometeram um reembolso há semanas e ainda não chegou.",
            "O suporte aprovou meu reembolso, mas nunca recebi.",
        ),
        "invoice_error": (
            "O nome da empresa e o CNPJ na minha nota estão errados.",
            "Minha fatura está com o endereço de cobrança errado.",
        ),
        "crash_on_save": (
            "O editor trava e fecha toda vez que salvo um rascunho.",
            "Salvar um documento faz o aplicativo fechar sozinho.",
        ),
        "sync_slow": (
            "As alterações demoram muito para aparecer nos meus outros aparelhos.",
            "A sincronização entre notebook e celular ficou lentíssima.",
        ),
        "export_broken": (
            "Ao exportar um relatório em PDF recebo um arquivo vazio.",
            "O botão de exportar fica girando e nada é baixado.",
        ),
        "images_missing": (
            "As imagens que envio aparecem como ícones quebrados.",
            "As fotos somem das minhas páginas depois de publicar.",
        ),
        "duplicate_notifications": (
            "Recebo cada notificação três ou quatro vezes.",
            "O aplicativo manda o mesmo alerta sem parar.",
        ),
        "password_reset": (
            "O e-mail para redefinir a senha nunca chega.",
            "Pedi um link de nova senha, mas não recebo nada.",
        ),
        "sso_error": (
            "O login único com a conta da empresa mostra uma página de erro.",
            "Quando entro pelo nosso provedor de identidade, volto para a tela de login.",
        ),
        "two_factor_lost": (
            "Perdi o celular com o app autenticador e não passo na verificação em duas etapas.",
            "Meus códigos de dois fatores pararam de funcionar depois que troquei de celular.",
        ),
        "account_locked": (
            "Minha conta foi bloqueada depois de algumas senhas erradas.",
            "O site diz que minha conta está bloqueada e não consigo entrar.",
        ),
        "invite_expired": (
            "O link de convite do meu colega diz que expirou.",
            "Não consigo entrar na equipe porque o convite não funciona mais.",
        ),
        "csv_export": (
            "Ajudaria muito poder baixar nossos dados em CSV.",
            "Vocês poderiam adicionar exportação para planilha?",
        ),
        "bulk_edit": (
            "Por favor, deixem alterar vários registros de uma vez, e não um por um.",
            "Adoraríamos editar um lote inteiro de itens juntos.",
        ),
        "dark_mode": (
            "Um tema escuro deixaria o app mais confortável para os olhos.",
            "Pensem em adicionar um modo noturno, por favor.",
        ),
        "api_webhooks": (
            "Queremos webhooks para nossos sistemas saberem das mudanças na hora.",
            "A API poderia enviar eventos para o nosso servidor?",
        ),
        "calendar_sync": (
            "Seria ótimo sincronizar os prazos com minha agenda.",
            "Adicionem uma opção para ver os vencimentos no Google ou Outlook.",
        ),
    },
    "cue": {
        4: (
            "Toda a nossa equipe está parada e a produção caiu.",
            "É uma queda para a empresa inteira.",
            "Nada funciona para nenhum usuário agora, é uma emergência.",
        ),
        3: (
            "Isso trava o meu trabalho hoje, meus colegas estão bem.",
            "Não consigo fazer meu trabalho até isso ser resolvido.",
            "Estou travado e não consigo continuar minhas tarefas.",
        ),
        2: (
            "É chato, mas por enquanto tenho um jeito de contornar.",
            "Me atrasa, mas existe uma alternativa.",
            "Existe um contorno meio desajeitado, então pode esperar um pouco.",
        ),
        1: (
            "Sem pressa nenhuma, nada está travado.",
            "Só uma observação de baixa prioridade, com calma.",
            "É só algo legal de ter, quando puderem.",
        ),
    },
    "tier": ("Estou no plano {tier}.", "(tipo de conta: {tier})"),
    "order": ("Pedido: {order}.", "É sobre o pedido {order}."),
    "refund": ("Por favor, reembolsem {amount} USD.", "Quero {amount} USD de volta."),
    "open": ("Oi, equipe,", "Olá,", "Bom dia, suporte,"),
    "close": ("Obrigado.", "Abraços.", "Desde já agradeço."),
}

L["hi"] = {
    "from": "प्रेषक",
    "subject": "विषय",
    "subjects": {
        "billing": "बिल में समस्या",
        "bug": "कुछ काम नहीं कर रहा",
        "access": "लॉगिन नहीं हो रहा",
        "feature_request": "सुझाव",
    },
    "body": {
        "double_charge": (
            "एक ही महीने के लिए मुझसे दो बार पैसे काटे गए।",
            "इस बार मेरे कार्ड पर दो एक जैसे भुगतान दिख रहे हैं।",
        ),
        "cancelled_still_charged": (
            "मैंने पिछले महीने सदस्यता रद्द कर दी थी, फिर भी पैसे कटे।",
            "रद्द करने के बाद भी कार्ड से एक और शुल्क कट गया।",
        ),
        "wrong_amount": (
            "जो राशि ली गई वह वेबसाइट पर दी गई कीमत से ज़्यादा है।",
            "मेरे प्लान की कीमत से अधिक बिल किया गया।",
        ),
        "refund_status": (
            "कई हफ़्ते पहले रिफ़ंड का वादा किया गया था, पर अभी तक नहीं आया।",
            "सपोर्ट ने रिफ़ंड मंज़ूर किया था लेकिन मुझे कभी मिला नहीं।",
        ),
        "invoice_error": (
            "मेरे इनवॉइस पर कंपनी का नाम और टैक्स नंबर गलत है।",
            "इनवॉइस में बिलिंग पता गलत लिखा है।",
        ),
        "crash_on_save": (
            "ड्राफ़्ट सेव करते ही एडिटर अटककर बंद हो जाता है।",
            "दस्तावेज़ सेव करने पर ऐप क्रैश हो जाता है।",
        ),
        "sync_slow": (
            "बदलाव मेरे दूसरे डिवाइस पर दिखने में बहुत देर लगती है।",
            "लैपटॉप और फ़ोन के बीच सिंक बहुत धीमा हो गया है।",
        ),
        "export_broken": (
            "रिपोर्ट को PDF में एक्सपोर्ट करने पर खाली फ़ाइल मिलती है।",
            "एक्सपोर्ट बटन घूमता रहता है और कुछ डाउनलोड नहीं होता।",
        ),
        "images_missing": (
            "अपलोड की गई तस्वीरें टूटे हुए आइकन की तरह दिखती हैं।",
            "पब्लिश करने के बाद मेरे पेजों से फ़ोटो गायब हो जाती हैं।",
        ),
        "duplicate_notifications": (
            "हर सूचना मुझे तीन-चार बार मिलती है।",
            "ऐप बार-बार वही अलर्ट भेजता रहता है।",
        ),
        "password_reset": (
            "पासवर्ड रीसेट का ईमेल कभी नहीं आता।",
            "मैंने नए पासवर्ड का लिंक माँगा पर कुछ नहीं आया।",
        ),
        "sso_error": (
            "कंपनी लॉगिन से सिंगल साइन-ऑन करने पर एरर पेज आता है।",
            "हमारे आइडेंटिटी प्रोवाइडर से लॉगिन करने पर मैं वापस लॉगिन स्क्रीन पर पहुँच जाता हूँ।",
        ),
        "two_factor_lost": (
            "ऑथेंटिकेटर ऐप वाला फ़ोन खो गया है और टू-फ़ैक्टर पार नहीं हो रहा।",
            "फ़ोन बदलने के बाद मेरे टू-फ़ैक्टर कोड काम नहीं कर रहे।",
        ),
        "account_locked": (
            "कुछ गलत पासवर्ड के बाद मेरा खाता लॉक हो गया।",
            "साइट कहती है कि मेरा खाता लॉक है और मैं लॉगिन नहीं कर पा रहा।",
        ),
        "invite_expired": (
            "सहकर्मी का भेजा आमंत्रण लिंक कहता है कि उसकी अवधि खत्म हो गई।",
            "आमंत्रण लिंक अब काम नहीं करता, इसलिए मैं टीम में शामिल नहीं हो पा रहा।",
        ),
        "csv_export": (
            "अगर हम अपना डेटा CSV में डाउनलोड कर सकें तो बहुत मदद होगी।",
            "क्या आप स्प्रेडशीट में एक्सपोर्ट का विकल्प जोड़ सकते हैं?",
        ),
        "bulk_edit": (
            "कृपया एक-एक करके नहीं, बल्कि कई रिकॉर्ड एक साथ बदलने दें।",
            "हमें आइटम के पूरे समूह को एक साथ एडिट करने का तरीका चाहिए।",
        ),
        "dark_mode": (
            "डार्क थीम से ऐप आँखों के लिए आरामदायक होगा।",
            "कृपया नाइट मोड जोड़ने पर विचार करें।",
        ),
        "api_webhooks": (
            "हम वेबहुक चाहते हैं ताकि हमारे सिस्टम को बदलाव तुरंत पता चलें।",
            "क्या API हमारे सर्वर पर इवेंट भेज सकता है?",
        ),
        "calendar_sync": (
            "डेडलाइन को मेरे कैलेंडर से सिंक कर पाना बढ़िया होगा।",
            "Google या Outlook कैलेंडर में ड्यू डेट दिखाने का विकल्प जोड़ें।",
        ),
    },
    "cue": {
        4: (
            "हमारी पूरी टीम रुकी हुई है और प्रोडक्शन बंद है।",
            "यह पूरी कंपनी के लिए आउटेज है।",
            "अभी किसी भी उपयोगकर्ता के लिए कुछ नहीं चल रहा, यह आपातकाल है।",
        ),
        3: (
            "इससे आज मेरा काम रुका है, हालाँकि सहकर्मी ठीक हैं।",
            "जब तक यह ठीक नहीं होता, मैं अपना काम नहीं कर सकता।",
            "मैं अटका हुआ हूँ और अपने काम आगे नहीं बढ़ा पा रहा।",
        ),
        2: (
            "परेशानी है, पर अभी के लिए एक वैकल्पिक तरीका है।",
            "इससे काम धीमा होता है, लेकिन एक जुगाड़ मौजूद है।",
            "एक अटपटा जुगाड़ है, इसलिए थोड़ा इंतज़ार कर सकते हैं।",
        ),
        1: (
            "कोई जल्दी नहीं, कुछ भी रुका नहीं है।",
            "बस कम प्राथमिकता वाली बात है, आराम से देखें।",
            "यह सिर्फ़ अच्छा-हो-तो-बेहतर है, जब समय मिले।",
        ),
    },
    "tier": ("मेरा प्लान {tier} है।", "(खाता प्रकार: {tier})"),
    "order": ("ऑर्डर: {order}।", "यह ऑर्डर {order} के बारे में है।"),
    "refund": ("कृपया {amount} USD वापस करें।", "मुझे {amount} USD वापस चाहिए।"),
    "open": ("नमस्ते टीम,", "नमस्कार,", "सुप्रभात सपोर्ट,"),
    "close": ("धन्यवाद।", "शुभकामनाएँ।", "पहले से धन्यवाद।"),
}

L["zh"] = {
    "from": "发件人",
    "subject": "主题",
    "subjects": {
        "billing": "账单问题",
        "bug": "功能出错",
        "access": "无法登录",
        "feature_request": "建议",
    },
    "body": {
        "double_charge": ("同一个月被扣了两次款。", "这个周期我的卡上有两笔一模一样的付款。"),
        "cancelled_still_charged": (
            "我上个月已经取消订阅，但还是被扣款了。",
            "取消之后卡上又出现了一笔扣费。",
        ),
        "wrong_amount": ("扣的金额比网站上的价格高。", "收费比我的套餐价格多。"),
        "refund_status": (
            "几周前答应给我退款，到现在还没到账。",
            "客服批准了退款，但我一直没收到。",
        ),
        "invoice_error": ("发票上的公司名称和税号写错了。", "发票上的账单地址不对。"),
        "crash_on_save": ("每次保存草稿，编辑器都会卡住然后关闭。", "一保存文档，应用就崩溃。"),
        "sync_slow": ("修改要很久才会出现在我的其他设备上。", "电脑和手机之间的同步变得非常慢。"),
        "export_broken": (
            "把报告导出成 PDF 得到的是空文件。",
            "导出按钮一直转圈，什么也下载不了。",
        ),
        "images_missing": ("上传的图片显示成破损图标。", "发布之后页面上的照片就不见了。"),
        "duplicate_notifications": ("每条通知我都会收到三四次。", "应用一遍又一遍地发同一个提醒。"),
        "password_reset": ("重置密码的邮件一直收不到。", "我申请了新密码链接，但什么都没收到。"),
        "sso_error": (
            "用公司账号单点登录时出现错误页面。",
            "通过我们的身份提供商登录后，又被跳回登录页。",
        ),
        "two_factor_lost": (
            "装有验证器应用的手机丢了，过不了双重验证。",
            "换手机以后，双重验证码就不能用了。",
        ),
        "account_locked": ("输错几次密码后账号被锁定了。", "网站提示我的账号已锁定，无法登录。"),
        "invite_expired": ("同事发来的邀请链接显示已过期。", "邀请链接失效了，我加入不了团队。"),
        "csv_export": ("如果能把数据下载成 CSV 会很有帮助。", "能否增加导出为表格的选项？"),
        "bulk_edit": ("请支持一次修改多条记录，而不是一条一条改。", "希望能一次批量编辑一组条目。"),
        "dark_mode": ("深色主题会让眼睛更舒服。", "请考虑增加夜间模式。"),
        "api_webhooks": (
            "我们希望有 webhook，让系统第一时间知道变更。",
            "API 能否把事件推送到我们的服务器？",
        ),
        "calendar_sync": (
            "如果能把截止日期同步到我的日历就太好了。",
            "请增加在 Google 或 Outlook 日历中显示到期日的选项。",
        ),
    },
    "cue": {
        4: (
            "我们整个团队都被卡住了，生产环境已经宕机。",
            "这是全公司范围的故障。",
            "现在所有用户都完全用不了，情况紧急。",
        ),
        3: (
            "这让我今天的工作无法进行，同事们倒没问题。",
            "问题不解决我就没法干活。",
            "我卡住了，手头的任务没法继续。",
        ),
        2: (
            "有点烦，不过目前有变通办法。",
            "会拖慢我，但有绕过的方法。",
            "有个笨办法可以凑合，所以可以稍等。",
        ),
        1: (
            "完全不着急，没有任何影响。",
            "只是低优先级的反馈，慢慢来。",
            "只是锦上添花，有空再说。",
        ),
    },
    "tier": ("我用的是 {tier} 套餐。", "（账户类型：{tier}）"),
    "order": ("订单：{order}。", "这和订单 {order} 有关。"),
    "refund": ("请退还 {amount} USD。", "我希望退回 {amount} USD。"),
    "open": ("你好，", "您好，", "客服早上好，"),
    "close": ("谢谢。", "祝好。", "先谢谢了。"),
}

L["ja"] = {
    "from": "差出人",
    "subject": "件名",
    "subjects": {
        "billing": "請求について",
        "bug": "不具合",
        "access": "ログインできない",
        "feature_request": "要望",
    },
    "body": {
        "double_charge": (
            "同じ月の料金が二回請求されました。",
            "今回、カードに同じ金額の支払いが二件あります。",
        ),
        "cancelled_still_charged": (
            "先月解約したのに、まだ引き落とされました。",
            "解約後にもう一度請求がありました。",
        ),
        "wrong_amount": (
            "請求額がサイトに載っている価格より高いです。",
            "プランの料金より多く請求されました。",
        ),
        "refund_status": (
            "数週間前に返金すると言われましたが、まだ届いていません。",
            "サポートが返金を承認したのに、受け取っていません。",
        ),
        "invoice_error": (
            "請求書の会社名と税番号が間違っています。",
            "請求書の請求先住所が違います。",
        ),
        "crash_on_save": (
            "下書きを保存するたびにエディタが固まって閉じます。",
            "文書を保存するとアプリが落ちます。",
        ),
        "sync_slow": (
            "変更がほかの端末に反映されるまでとても時間がかかります。",
            "パソコンとスマホの同期が極端に遅くなりました。",
        ),
        "export_broken": (
            "レポートを PDF に書き出すと空のファイルになります。",
            "エクスポートボタンが回り続けて何もダウンロードされません。",
        ),
        "images_missing": (
            "アップロードした画像が壊れたアイコンで表示されます。",
            "公開したあとページから写真が消えます。",
        ),
        "duplicate_notifications": (
            "通知が毎回三、四回届きます。",
            "アプリが同じアラートを何度も送ってきます。",
        ),
        "password_reset": (
            "パスワード再設定のメールが届きません。",
            "新しいパスワードのリンクを頼みましたが何も来ません。",
        ),
        "sso_error": (
            "会社アカウントのシングルサインオンでエラーページが出ます。",
            "ID プロバイダー経由でログインするとログイン画面に戻されます。",
        ),
        "two_factor_lost": (
            "認証アプリの入ったスマホをなくして二段階認証を通れません。",
            "機種変更してから二段階認証のコードが使えません。",
        ),
        "account_locked": (
            "パスワードを何回か間違えたらアカウントがロックされました。",
            "アカウントがロックされているとサイトに表示され、ログインできません。",
        ),
        "invite_expired": (
            "同僚からの招待リンクが期限切れと表示されます。",
            "招待リンクが使えなくなり、チームに参加できません。",
        ),
        "csv_export": (
            "データを CSV でダウンロードできると助かります。",
            "表計算ソフト向けのエクスポートを追加できますか？",
        ),
        "bulk_edit": (
            "一件ずつではなく、まとめて多くのレコードを変更できるようにしてください。",
            "複数の項目を一括で編集できる機能がほしいです。",
        ),
        "dark_mode": (
            "ダークテーマがあれば目が楽になります。",
            "夜間モードの追加を検討してください。",
        ),
        "api_webhooks": (
            "変更をすぐ受け取れるように webhook がほしいです。",
            "API からこちらのサーバーにイベントを送れますか？",
        ),
        "calendar_sync": (
            "締め切りをカレンダーと同期できると最高です。",
            "Google や Outlook のカレンダーに期日を表示するオプションを追加してください。",
        ),
    },
    "cue": {
        4: (
            "チーム全体が止まっていて、本番環境がダウンしています。",
            "会社全体で障害が起きています。",
            "今はどのユーザーも何も使えず、緊急事態です。",
        ),
        3: (
            "今日の私の作業が止まっています。同僚は問題ありません。",
            "これが直るまで仕事ができません。",
            "行き詰まっていて作業を進められません。",
        ),
        2: (
            "困っていますが、今のところ回避策があります。",
            "作業は遅くなりますが、回避方法はあります。",
            "不格好な回避策があるので、少し待てます。",
        ),
        1: (
            "急ぎではありません。何も止まっていません。",
            "優先度の低い連絡です。ごゆっくりどうぞ。",
            "あったら嬉しい程度なので、手が空いたときで大丈夫です。",
        ),
    },
    "tier": ("{tier} プランを使っています。", "（アカウント種別：{tier}）"),
    "order": ("注文：{order}。", "注文 {order} の件です。"),
    "refund": ("{amount} USD を返金してください。", "{amount} USD を返してほしいです。"),
    "open": ("お世話になっております。", "こんにちは。", "サポートご担当者様"),
    "close": (
        "よろしくお願いします。",
        "ありがとうございます。",
        "お手数ですがよろしくお願いいたします。",
    ),
}

LANGUAGES: tuple[str, ...] = tuple(L)

# Chatter in a second language, for mixed-language noise.
FOREIGN_CHATTER: dict[str, tuple[str, ...]] = {
    "en": ("Sorry, mi inglés no es perfecto.", "Merci beaucoup!"),
    "other": ("Sorry for my English, I hope this is clear.", "Sent from my phone."),
}

# Extra chatter that carries no field values. The third entry is test-only.
EXTRA_CHATTER: dict[str, tuple[str, ...]] = {
    "en": (
        "I already tried logging out and back in.",
        "My colleague says it worked fine last week.",
        "I have been a customer for years and never had this before.",
    ),
    "other": (
        "PS: I already restarted the app.",
        "PS: same thing on the web version.",
        "PS: a colleague sees it too.",
    ),
}

EMAILS: tuple[str, ...] = ("user@example.test", "ops@example.test", "help-me@example.test")
