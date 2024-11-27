;; This software is courtesy @http://quil.info/, Erik Svedäng
;; --------------------------------------------
;; http://quil.info/sketches/show/example_waves
;; --------------------------------------------

(ns ftlm.disco.wave
  (:require
   [quil.core :as q :include-macros true]
   [ftlm.vehicles.art.lib :refer [*dt*] :as lib]
   [ftlm.vehicles.art.defs :as defs]))

(defn setup []
  (q/frame-rate 30))

(defonce goes-left? (atom false))

(defn t
  []
  (* 0.001 (q/millis))
  ;; (when (< 0.999
  ;;          (q/random 1))
  ;;   (swap! goes-left? not))
  ;; (* (if @goes-left? 1 -1) 1 (* 0.001 (q/millis)))
  )

(defn calc-y [x mid amp]
  (+ mid (* (q/sin (+ (t) x)) amp)))

(defn wave
  [step mid-y amp]
  (q/push-matrix)
  (q/with-translation
    [0 500]
    (let [w (q/width)
          h 200
          mult (q/map-range w 800 100 0.01 0.5)]
      (q/begin-shape)
      (q/vertex 0 h)
      (doseq [x (range
                 (- w)
                 (+ step w)
                 step)]
        (let [t (* x mult)
              y (calc-y t mid-y amp)]
          (q/vertex x y)))
      (q/vertex w h)
      (q/end-shape)))
  (q/pop-matrix))

(defn draw
  []
  ;; (q/background 250)
  (q/stroke 255 250)
  ;; (if (< 0.2 (q/random 1))
  ;;   (q/stroke 0 0))
  (q/stroke-weight 10)

  ;; (if (< 0.2 (q/random 1))
  ;;   ;; (q/fill 50 230 (+ (* 20 (q/sin (* 20 (t)))) 230) 40)
  ;;   (q/fill
  ;;    (lib/->hsb (:red defs/color-map)))
  ;;   (q/fill
  ;;    (lib/->hsb (:cyan defs/color-map)))
  ;;   ;; (q/fill 0 0 0)
  ;;   )
  (let [h 200
        move-down 5
        amp 50
        ;; (rand-nth [0 20 100])
        ]
    (doseq [y (range move-down (+ amp h) 5)]
      (let [x-step (- (* y 0.8) move-down)]
        (wave x-step y amp)))))





;; (q/defsketch waves
;;    :host "host"
;;    :size [500 500]
;;    :setup setup
;;   :draw draw)
