(ns ftlm.disco.disco-middleware
  (:require [quil.core :as q :include-macros true]))

(defn default-position
  "Default position configuration: zoom is neutral and central point is
  `width/2, height/2`."
  []
  {:position [(/ (q/width) 2.0)
              (/ (q/height) 2.0)]
   :zoom 1
   :mouse-buttons #{:left :right :center}})


(defn picture-jitter
  [options]
  (let [user-settings (:picture-jitter options)
        user-draw (:draw options (fn [state]))
        setup (:setup options (fn [] {}))

        ]

    )


  )
