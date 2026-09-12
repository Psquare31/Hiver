"""My hand labels for the 210 AppleSupport golden examples.

Produced by reading every message against config/taxonomy.yaml. Recorded as
source rather than a bare CSV so the reasoning is reviewable and diffable.

ROUTE REASON CODES
  Escalate:
    safety_hazard         physical risk - swelling, overheating, shock, injury
    data_loss             irreplaceable data reported missing
    account_security      identity or Apple ID ownership at stake
    money_dispute         disputed charge, refund, or purchase problem
    account_lookup        needs account-specific data a public bot cannot see
    service_booking       needs repair, warranty, replacement or an appointment
    repeat_contact        customer states they have already been failed
    churn_risk            explicit threat to leave the platform
    human_requested       customer explicitly asks for a person
  Auto:
    troubleshooting       standard diagnostic steps apply
    informational         answerable from public product facts
    known_issue           a documented bug with a standard public response
    feature_feedback      product opinion or feature request
    acknowledgement       thanks, praise, or closing; no action needed
    no_action             contentless or off-topic

BOUNDARY CALLS, recorded because a second annotator could reasonably differ and
these drive most of the disagreement:

  * DEVICE_PERFORMANCE vs APP_SOFTWARE. A named app misbehaving (Safari, Pages,
    Music, Messages, keyboard) is app_software; the whole device freezing,
    crashing or slowing is device_performance. This is the largest source of
    near-misses in the set.
  * The iOS 11 "I" -> boxed-question-mark bug (g000, g005, g010, g012, g013,
    g020, g021, g026, g030, g031, g036, g045, g048, g057, g075, g108, g140,
    g146, g147, g158, g159, g160, g170, g180, g186, g188, g194, g202, g203) is
    app_software with reason `known_issue`. It is ~14% of the whole sample. That
    single 2017 bug is a large slice of this corpus and is called out in the
    report - a system tuned on this data is partly tuned on one bug.
  * BATTERY vs DEVICE_PERFORMANCE when both appear: battery wins, because the
    remedy differs and battery is the more consequential complaint.
  * HARDWARE_REPAIR vs DEVICE_PERFORMANCE: physical damage, a failed component
    (mic, speaker, camera), or anything already through a service centre is
    hardware_repair, because the next step is a booking rather than a step list.
  * Non-English messages (g063, g126, g133, g150, g182, g201) are labelled by
    their actual intent, not dumped in `other`.

RULE GAPS FOUND WHILE LABELLING - the escalation regexes in taxonomy.yaml do
NOT catch these, and I labelled what a careful human would decide, not what the
rules produce. The difference is a real measured failure mode, not an oversight:
  * g067 "practically on fire"  - trigger list has "caught fire", not "on fire"
  * g145 "electric shock"       - not in the safety list at all
  * g154 "electric shock ... blister" - actual injury, not matched
  * g191 "not supposed to inflate" - trigger list has "swelling"/"bulging"
  * g123 "WHERE ARE MY PHOTOS"  - data_loss list expects "photos are gone"
  * g085 "Third time today my phone freezes" - repeat_contact FALSE POSITIVE:
    "third time" here counts crashes, not support contacts. Labelled auto.
"""

# golden_id: (intent, route, reason_code)
LABELS: dict[str, tuple[str, str, str]] = {
    "g000": ("app_software", "auto", "known_issue"),
    "g001": ("battery_power", "auto", "troubleshooting"),
    "g002": ("app_software", "auto", "known_issue"),
    "g003": ("os_update", "auto", "troubleshooting"),
    "g004": ("device_performance", "auto", "troubleshooting"),
    "g005": ("app_software", "auto", "known_issue"),
    "g006": ("os_update", "auto", "informational"),
    "g007": ("app_software", "auto", "troubleshooting"),
    "g008": ("acknowledgement", "auto", "acknowledgement"),
    "g009": ("other", "auto", "informational"),
    "g010": ("app_software", "auto", "known_issue"),
    "g011": ("connectivity", "auto", "troubleshooting"),
    "g012": ("app_software", "auto", "known_issue"),
    "g013": ("app_software", "auto", "known_issue"),
    "g014": ("device_performance", "auto", "troubleshooting"),
    "g015": ("other", "auto", "informational"),
    "g016": ("app_software", "auto", "informational"),
    "g017": ("device_performance", "auto", "troubleshooting"),
    "g018": ("battery_power", "auto", "troubleshooting"),
    "g019": ("battery_power", "escalate", "safety_hazard"),
    "g020": ("app_software", "auto", "known_issue"),
    "g021": ("app_software", "auto", "known_issue"),
    "g022": ("hardware_repair", "escalate", "service_booking"),
    "g023": ("feedback_complaint", "auto", "feature_feedback"),
    "g024": ("battery_power", "escalate", "repeat_contact"),
    "g025": ("battery_power", "auto", "troubleshooting"),
    "g026": ("app_software", "auto", "known_issue"),
    "g027": ("device_performance", "auto", "troubleshooting"),
    "g028": ("device_performance", "auto", "troubleshooting"),
    "g029": ("os_update", "auto", "troubleshooting"),
    "g030": ("app_software", "auto", "known_issue"),
    "g031": ("app_software", "auto", "known_issue"),
    "g032": ("connectivity", "auto", "troubleshooting"),
    "g033": ("app_software", "auto", "troubleshooting"),
    "g034": ("device_performance", "auto", "troubleshooting"),
    "g035": ("connectivity", "auto", "troubleshooting"),
    "g036": ("app_software", "auto", "known_issue"),
    "g037": ("battery_power", "auto", "troubleshooting"),
    "g038": ("battery_power", "auto", "troubleshooting"),
    "g039": ("battery_power", "auto", "troubleshooting"),
    "g040": ("os_update", "auto", "troubleshooting"),
    "g041": ("app_software", "auto", "troubleshooting"),
    "g042": ("account_appleid", "escalate", "money_dispute"),
    "g043": ("feedback_complaint", "auto", "feature_feedback"),
    "g044": ("device_performance", "auto", "troubleshooting"),
    "g045": ("app_software", "auto", "known_issue"),
    "g046": ("device_performance", "auto", "troubleshooting"),
    "g047": ("device_performance", "auto", "troubleshooting"),
    "g048": ("app_software", "auto", "known_issue"),
    "g049": ("device_performance", "auto", "troubleshooting"),
    "g050": ("battery_power", "auto", "troubleshooting"),
    "g051": ("feedback_complaint", "auto", "feature_feedback"),
    "g052": ("acknowledgement", "auto", "acknowledgement"),
    "g053": ("account_appleid", "escalate", "account_security"),
    "g054": ("app_software", "auto", "troubleshooting"),
    "g055": ("app_software", "auto", "troubleshooting"),
    "g056": ("feedback_complaint", "auto", "feature_feedback"),
    "g057": ("app_software", "auto", "known_issue"),
    "g058": ("connectivity", "auto", "troubleshooting"),
    "g059": ("app_software", "auto", "troubleshooting"),
    "g060": ("battery_power", "auto", "troubleshooting"),
    "g061": ("app_software", "auto", "troubleshooting"),
    "g062": ("device_performance", "auto", "troubleshooting"),
    "g063": ("app_software", "auto", "troubleshooting"),
    "g064": ("device_performance", "auto", "troubleshooting"),
    "g065": ("feedback_complaint", "auto", "feature_feedback"),
    "g066": ("app_software", "auto", "known_issue"),
    "g067": ("hardware_repair", "escalate", "safety_hazard"),
    "g068": ("os_update", "escalate", "repeat_contact"),
    "g069": ("other", "auto", "no_action"),
    "g070": ("other", "auto", "no_action"),
    "g071": ("device_performance", "auto", "troubleshooting"),
    "g072": ("battery_power", "auto", "troubleshooting"),
    "g073": ("device_performance", "auto", "troubleshooting"),
    "g074": ("device_performance", "auto", "troubleshooting"),
    "g075": ("app_software", "auto", "known_issue"),
    "g076": ("device_performance", "auto", "troubleshooting"),
    "g077": ("hardware_repair", "escalate", "service_booking"),
    "g078": ("other", "auto", "no_action"),
    "g079": ("hardware_repair", "escalate", "service_booking"),
    "g080": ("account_appleid", "escalate", "account_lookup"),
    "g081": ("device_performance", "auto", "troubleshooting"),
    "g082": ("feedback_complaint", "auto", "feature_feedback"),
    "g083": ("battery_power", "auto", "troubleshooting"),
    "g084": ("account_appleid", "escalate", "account_lookup"),
    "g085": ("device_performance", "auto", "troubleshooting"),
    "g086": ("battery_power", "auto", "troubleshooting"),
    "g087": ("hardware_repair", "escalate", "service_booking"),
    "g088": ("device_performance", "auto", "troubleshooting"),
    "g089": ("os_update", "auto", "troubleshooting"),
    "g090": ("battery_power", "auto", "troubleshooting"),
    "g091": ("hardware_repair", "escalate", "repeat_contact"),
    "g092": ("app_software", "auto", "troubleshooting"),
    "g093": ("other", "escalate", "account_security"),
    "g094": ("connectivity", "auto", "troubleshooting"),
    "g095": ("app_software", "escalate", "repeat_contact"),
    "g096": ("feedback_complaint", "auto", "feature_feedback"),
    "g097": ("device_performance", "auto", "troubleshooting"),
    "g098": ("device_performance", "auto", "troubleshooting"),
    "g099": ("app_software", "auto", "informational"),
    "g100": ("battery_power", "auto", "troubleshooting"),
    "g101": ("app_software", "auto", "known_issue"),
    "g102": ("account_appleid", "escalate", "money_dispute"),
    "g103": ("app_software", "auto", "troubleshooting"),
    "g104": ("account_appleid", "escalate", "account_lookup"),
    "g105": ("connectivity", "auto", "troubleshooting"),
    "g106": ("device_performance", "auto", "troubleshooting"),
    "g107": ("feedback_complaint", "escalate", "repeat_contact"),
    "g108": ("app_software", "escalate", "churn_risk"),
    "g109": ("app_software", "auto", "troubleshooting"),
    "g110": ("device_performance", "auto", "troubleshooting"),
    "g111": ("hardware_repair", "escalate", "service_booking"),
    "g112": ("feedback_complaint", "auto", "feature_feedback"),
    "g113": ("device_performance", "auto", "troubleshooting"),
    "g114": ("device_performance", "auto", "troubleshooting"),
    "g115": ("device_performance", "auto", "troubleshooting"),
    "g116": ("other", "escalate", "repeat_contact"),
    "g117": ("hardware_repair", "escalate", "service_booking"),
    "g118": ("battery_power", "auto", "troubleshooting"),
    "g119": ("feedback_complaint", "auto", "feature_feedback"),
    "g120": ("battery_power", "auto", "troubleshooting"),
    "g121": ("connectivity", "auto", "troubleshooting"),
    "g122": ("device_performance", "auto", "troubleshooting"),
    "g123": ("account_appleid", "escalate", "data_loss"),
    "g124": ("account_appleid", "escalate", "data_loss"),
    "g125": ("connectivity", "escalate", "repeat_contact"),
    "g126": ("battery_power", "auto", "troubleshooting"),
    "g127": ("hardware_repair", "escalate", "service_booking"),
    "g128": ("device_performance", "auto", "troubleshooting"),
    "g129": ("other", "escalate", "service_booking"),
    "g130": ("acknowledgement", "auto", "acknowledgement"),
    "g131": ("other", "escalate", "repeat_contact"),
    "g132": ("hardware_repair", "escalate", "service_booking"),
    "g133": ("account_appleid", "escalate", "data_loss"),
    "g134": ("os_update", "auto", "informational"),
    "g135": ("device_performance", "auto", "troubleshooting"),
    "g136": ("connectivity", "auto", "troubleshooting"),
    "g137": ("os_update", "auto", "troubleshooting"),
    "g138": ("app_software", "auto", "troubleshooting"),
    "g139": ("device_performance", "auto", "troubleshooting"),
    "g140": ("app_software", "auto", "known_issue"),
    "g141": ("app_software", "auto", "troubleshooting"),
    "g142": ("device_performance", "auto", "troubleshooting"),
    "g143": ("app_software", "auto", "troubleshooting"),
    "g144": ("hardware_repair", "escalate", "service_booking"),
    "g145": ("hardware_repair", "escalate", "safety_hazard"),
    "g146": ("app_software", "auto", "known_issue"),
    "g147": ("app_software", "auto", "known_issue"),
    "g148": ("hardware_repair", "escalate", "service_booking"),
    "g149": ("hardware_repair", "escalate", "service_booking"),
    "g150": ("account_appleid", "auto", "informational"),
    "g151": ("battery_power", "escalate", "service_booking"),
    "g152": ("os_update", "auto", "informational"),
    "g153": ("app_software", "auto", "troubleshooting"),
    "g154": ("hardware_repair", "escalate", "safety_hazard"),
    "g155": ("app_software", "auto", "troubleshooting"),
    "g156": ("os_update", "auto", "troubleshooting"),
    "g157": ("battery_power", "auto", "troubleshooting"),
    "g158": ("app_software", "auto", "known_issue"),
    "g159": ("app_software", "auto", "known_issue"),
    "g160": ("app_software", "auto", "known_issue"),
    "g161": ("device_performance", "escalate", "service_booking"),
    "g162": ("feedback_complaint", "escalate", "repeat_contact"),
    "g163": ("battery_power", "auto", "troubleshooting"),
    "g164": ("account_appleid", "escalate", "account_lookup"),
    "g165": ("device_performance", "auto", "troubleshooting"),
    "g166": ("account_appleid", "auto", "troubleshooting"),
    "g167": ("other", "auto", "no_action"),
    "g168": ("app_software", "auto", "troubleshooting"),
    "g169": ("app_software", "auto", "troubleshooting"),
    "g170": ("app_software", "auto", "known_issue"),
    "g171": ("other", "escalate", "account_security"),
    "g172": ("hardware_repair", "escalate", "service_booking"),
    "g173": ("connectivity", "auto", "troubleshooting"),
    "g174": ("connectivity", "auto", "troubleshooting"),
    "g175": ("device_performance", "auto", "troubleshooting"),
    "g176": ("other", "escalate", "human_requested"),
    "g177": ("connectivity", "auto", "troubleshooting"),
    "g178": ("battery_power", "auto", "troubleshooting"),
    "g179": ("device_performance", "auto", "troubleshooting"),
    "g180": ("app_software", "auto", "known_issue"),
    "g181": ("account_appleid", "escalate", "money_dispute"),
    "g182": ("other", "escalate", "account_lookup"),
    "g183": ("account_appleid", "escalate", "account_lookup"),
    "g184": ("hardware_repair", "auto", "informational"),
    "g185": ("battery_power", "auto", "troubleshooting"),
    "g186": ("app_software", "auto", "known_issue"),
    "g187": ("account_appleid", "escalate", "data_loss"),
    "g188": ("app_software", "auto", "known_issue"),
    "g189": ("other", "auto", "no_action"),
    "g190": ("other", "auto", "no_action"),
    "g191": ("hardware_repair", "escalate", "safety_hazard"),
    "g192": ("hardware_repair", "escalate", "service_booking"),
    "g193": ("device_performance", "auto", "troubleshooting"),
    "g194": ("app_software", "auto", "known_issue"),
    "g195": ("device_performance", "auto", "troubleshooting"),
    "g196": ("connectivity", "auto", "troubleshooting"),
    "g197": ("connectivity", "auto", "troubleshooting"),
    "g198": ("connectivity", "escalate", "churn_risk"),
    "g199": ("feedback_complaint", "auto", "feature_feedback"),
    "g200": ("feedback_complaint", "escalate", "churn_risk"),
    "g201": ("app_software", "auto", "troubleshooting"),
    "g202": ("app_software", "auto", "known_issue"),
    "g203": ("app_software", "auto", "known_issue"),
    "g204": ("connectivity", "escalate", "repeat_contact"),
    "g205": ("feedback_complaint", "escalate", "churn_risk"),
    "g206": ("other", "escalate", "service_booking"),
    "g207": ("connectivity", "escalate", "churn_risk"),
    "g208": ("device_performance", "auto", "troubleshooting"),
    "g209": ("app_software", "auto", "troubleshooting"),
}
