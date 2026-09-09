"""Hand-labelled agent replies for tuning and testing the substance classifier.

Labelled by reading the text and asking one question: *if a model had to draft
this reply, is there anything here to ground it on beyond "go away"?*

A credential request ("DM us your account email") is HANDOFF, not substance.
It is account-specific triage that moves the conversation to a private channel;
nothing about it can be reused to answer the next similar customer in public.
That distinction is the whole point of the metric, and it is the one a naive
keyword matcher gets wrong.
"""

# (text, expected_substantive)
LABELLED = [
    # --- substantive: diagnostic probes -----------------------------------
    ("Hey Justin, that's not cool! Can you let us know what device, operating "
     "system, and Spotify version you're using? /HT", True),
    ("Did the issue happen even after this option was enabled? If so, what "
     "device, operating system, and Spotify version are you using?", True),
    ("Has it always been this way? Can you let us know if it started after an "
     "app or OS X update?", True),
    ("Could you send us a screenshot of what you see when you click the link "
     "we provided? We'll see what we can suggest", True),
    ("What browser are you using, and does the same thing happen in a private "
     "window?", True),

    # --- substantive: instructions ----------------------------------------
    ("Try logging out and back in, then restart the app. That usually clears "
     "it up /AM", True),
    ("Head to Settings > Storage and clear the cache, then reinstall. Let us "
     "know how you get on", True),
    ("Make sure Offline Mode is turned off in Settings - that's a common cause "
     "of tracks greying out", True),

    # --- handoff: credential requests (the trap) --------------------------
    ("Hey Ciara! Could you DM us your account's email address? We'll take a "
     "look backstage /RH", False),
    ("Hey there! Help's here. Can you DM us your account's email address or "
     "username? We'll take a look backstage", False),
    ("Thanks for letting us know. Could you send us a DM with your account's "
     "email address or username? We'll take a look", False),
    ("Hey Harry. Can you DM us your account's username and email address? "
     "We'll take a look backstage /GK", False),

    # --- handoff: generic redirects ---------------------------------------
    ("Apologies for the unpleasant experience. Kindly share your details here: "
     "https://t.co/abc and I'll contact you. ^GS", False),
    ("I'm sorry about the delay with the installation. Kindly reach out to our "
     "team here: https://t.co/xyz (1/2) ^GK", False),
    ("My apologies. Please fill this form: https://t.co/qqq and I'll contact "
     "you at the earliest. ^NK", False),
    ("Hi! We've just sent a DM your way. Let's carry on chatting there /DR", False),
    ("We'd like to make sure this is escalated for you. Please send us your "
     "details here: https://t.co/z ^MO", False),

    # --- neither: sympathy / policy statements ----------------------------
    ("We're sorry you feel that way. Sometimes content gets removed due to "
     "licensing and legal issues.", False),
    ("Anytime! Let us know if you ever need us again. We'd be... /AN", False),
    ("Hey there! We're afraid there's no change log for the Android app, but "
     "we'll let the right team know it's wanted", False),
]
