"""Short support-ticket texts in several languages.

Each string includes the ticket id, the tier word, and, when the scenario
has them, the order reference and the refund amount. Category names are
not spelled out, so the label is not a copy of an enum token.
"""

TEMPLATES: dict[str, dict[str, str]] = {
    "en": {
        "billing": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: charged twice\n\n"
            "Hello, I was charged twice for order {order} on the {tier} plan. "
            "Please refund {amount} USD."
        ),
        "bug": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: crash on save\n\n"
            "The editor crashes when I save a draft. Account tier: {tier}. "
            "Related order {order}. I am not asking for money back."
        ),
        "access": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: cannot sign in\n\n"
            "I cannot sign in to the {tier} account. The reset mail never arrives."
        ),
        "feature_request": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: csv export\n\n"
            "Please add a CSV export for the {tier} plan."
        ),
    },
    "es": {
        "billing": (
            "Ticket: {tid}\nDe: user@example.test\nAsunto: cobro duplicado\n\n"
            "Hola, me cobraron dos veces el pedido {order} del plan {tier}. "
            "Necesito un reembolso de {amount} USD."
        ),
        "bug": (
            "Ticket: {tid}\nDe: user@example.test\nAsunto: error al guardar\n\n"
            "El editor se cierra al guardar un borrador. Plan: {tier}. "
            "Pedido relacionado {order}. No pido dinero de vuelta."
        ),
        "access": (
            "Ticket: {tid}\nDe: user@example.test\nAsunto: no puedo entrar\n\n"
            "No puedo entrar a la cuenta {tier}. El correo de restablecimiento no llega."
        ),
        "feature_request": (
            "Ticket: {tid}\nDe: user@example.test\nAsunto: exportar csv\n\n"
            "Por favor agreguen exportacion CSV para el plan {tier}."
        ),
    },
    "fr": {
        "billing": (
            "Ticket: {tid}\nDe: user@example.test\nObjet: double debit\n\n"
            "Bonjour, la commande {order} du plan {tier} a ete debitee deux fois. "
            "Merci de rembourser {amount} USD."
        ),
        "bug": (
            "Ticket: {tid}\nDe: user@example.test\nObjet: plantage\n\n"
            "L editeur se ferme quand j enregistre un brouillon. Plan: {tier}. "
            "Commande liee {order}. Je ne demande pas d argent."
        ),
        "access": (
            "Ticket: {tid}\nDe: user@example.test\nObjet: connexion impossible\n\n"
            "Je ne peux pas ouvrir le compte {tier}. Le message de reinitialisation n arrive pas."
        ),
        "feature_request": (
            "Ticket: {tid}\nDe: user@example.test\nObjet: export csv\n\n"
            "Merci d ajouter un export CSV pour le plan {tier}."
        ),
    },
    "de": {
        "billing": (
            "Ticket: {tid}\nVon: user@example.test\nBetreff: doppelte Abbuchung\n\n"
            "Hallo, die Bestellung {order} im Tarif {tier} wurde zweimal abgebucht. "
            "Bitte erstatten Sie {amount} USD."
        ),
        "bug": (
            "Ticket: {tid}\nVon: user@example.test\nBetreff: Absturz\n\n"
            "Der Editor stuerzt beim Speichern ab. Tarif: {tier}. "
            "Zugehoerige Bestellung {order}. Ich will kein Geld zurueck."
        ),
        "access": (
            "Ticket: {tid}\nVon: user@example.test\nBetreff: Anmeldung\n\n"
            "Ich kann mich im Konto {tier} nicht anmelden. Die Zuruecksetzen-Mail kommt nicht an."
        ),
        "feature_request": (
            "Ticket: {tid}\nVon: user@example.test\nBetreff: CSV-Export\n\n"
            "Bitte einen CSV-Export fuer den Tarif {tier} hinzufuegen."
        ),
    },
    "pt": {
        "billing": (
            "Ticket: {tid}\nDe: user@example.test\nAssunto: cobranca duplicada\n\n"
            "Ola, cobraram duas vezes o pedido {order} do plano {tier}. "
            "Preciso do reembolso de {amount} USD."
        ),
        "bug": (
            "Ticket: {tid}\nDe: user@example.test\nAssunto: falha ao salvar\n\n"
            "O editor fecha ao salvar um rascunho. Plano: {tier}. "
            "Pedido relacionado {order}. Nao peco dinheiro de volta."
        ),
        "access": (
            "Ticket: {tid}\nDe: user@example.test\nAssunto: nao consigo entrar\n\n"
            "Nao consigo entrar na conta {tier}. O email de redefinicao nao chega."
        ),
        "feature_request": (
            "Ticket: {tid}\nDe: user@example.test\nAssunto: exportar csv\n\n"
            "Por favor adicionem exportacao CSV para o plano {tier}."
        ),
    },
    "ja": {
        "billing": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: double charge\n\n"
            "注文 {order}（{tier} プラン）が二回請求されました。{amount} USD を返金してください。"
        ),
        "bug": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: crash\n\n"
            "下書きを保存するとエディタが落ちます。プラン: {tier}。"
            "関連注文 {order}。返金は求めていません。"
        ),
        "access": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: sign in\n\n"
            "{tier} アカウントにサインインできません。リセットメールが届きません。"
        ),
        "feature_request": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: csv\n\n"
            "{tier} プランに CSV 書き出しを追加してください。"
        ),
    },
    "hi": {
        "billing": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: double charge\n\n"
            "ऑर्डर {order} ({tier} प्लान) का शुल्क दो बार कटा। कृपया {amount} USD वापस करें।"
        ),
        "bug": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: crash\n\n"
            "ड्राफ्ट सेव करते ही एडिटर बंद हो जाता है। प्लान: {tier}। "
            "संबंधित ऑर्डर {order}। पैसे वापस नहीं चाहिए।"
        ),
        "access": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: sign in\n\n"
            "{tier} खाते में साइन इन नहीं हो पा रहा। रीसेट मेल नहीं आ रहा।"
        ),
        "feature_request": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: csv\n\n"
            "{tier} प्लान के लिए CSV निर्यात जोड़ें।"
        ),
    },
    "zh": {
        "billing": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: double charge\n\n"
            "订单 {order}（{tier} 方案）被扣了两次。请退回 {amount} USD。"
        ),
        "bug": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: crash\n\n"
            "保存草稿时编辑器会退出。方案: {tier}。相关订单 {order}。不需要退款。"
        ),
        "access": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: sign in\n\n"
            "无法登录 {tier} 账户。重置邮件没有收到。"
        ),
        "feature_request": (
            "Ticket: {tid}\nFrom: user@example.test\nSubject: csv\n\n"
            "请为 {tier} 方案增加 CSV 导出。"
        ),
    },
}

LANGUAGES: tuple[str, ...] = tuple(TEMPLATES)
SCENARIOS: tuple[str, ...] = ("billing", "bug", "access", "feature_request")
