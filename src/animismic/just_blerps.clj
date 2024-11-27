(ns animismic.f
  (:require
    [libpython-clj2.require :refer [require-python]]
    [libpython-clj2.python :refer [py. py..] :as py]
    [animismic.lib.particles :as p]
    [fastmath.core :as fm]
    [clojure.java.io :as io]
    [clojure.string :as str]
    [clojure.data.json :as json]
    [quil.middleware :as m]
    [ftlm.vehicles.art.lib :refer [*dt*] :as libjp]
    [ftlm.vehicles.art.defs :as defs]
    [ftlm.vehicles.art.extended :as elib]
    [tech.v3.datatype.unary-pred :as unary-pred]
    [tech.v3.datatype.functional :as f]
    [tech.v3.datatype :as dtype]
    [tech.v3.tensor :as dtt]
    [fastmath.random :as fm.rand]
    [quil.core :as q]
    [tech.v3.datatype.functional :as f]
    [tech.v3.datatype :as dtype]
    [tech.v3.tensor :as dtt]
    [tech.v3.datatype.bitmap :as bitmap]
    [fastmath.random :as fm.rand]
    [bennischwerdtner.hd.binary-sparse-segmented :as hd]
    [bennischwerdtner.pyutils :as pyutils]
    [tech.v3.datatype.unary-pred :as unary-pred]
    [tech.v3.datatype.argops :as dtype-argops]
    [bennischwerdtner.sdm.sdm :as sdm]
    [bennischwerdtner.hd.codebook-item-memory :as codebook]
    [bennischwerdtner.hd.ui.audio :as audio]
    [bennischwerdtner.hd.data :as hdd]
    [animismic.lib.blerp :as b]
    [ftlm.disco.middlewares.picture-jitter :as
     picture-jitter]
    [ftlm.vehicles.art.event-q-middleware :as evtq]
    [ftlm.disco.middlewares.time-warp :as time-warp]))

;; __________________

(def grid-width 30)

;;
(def alphabet (into [] (range 2)))

(def letter->color
  [0
   (defs/color-map :cyan)
   (defs/color-map :green-yellow)
   (defs/color-map :magenta)])

;; --------------------

;; (defonce gluespace)

(defn blerp-glue-space
  []
  (let [sdm (sdm/->sdm {:address-count (long 1e6)
                        :address-density 0.000003
                        :word-length (long 1e4)})]))



(defn update-blerp
  [e s _]
  ;; (hd/unbind b/berp-map (:particle-id e))
  (when-let [world (first (filter :world? (lib/entities s)))]
    (let [ ;; berp-id -> alphabet
          ;;
          factor
          (get
           {:heliotrope 0 :orange 0}
           (:particle-id e))
          ;; (b/blerp-resonator-force (:particle-id e)
          ;;                          (:blerp-map glooby))
          world-activations (:elements world)
          letter 1]
      ;; ------------------------------------------
      (update-in e
                 [:particle-field]
                 p/resonate-update
                 world-activations
                 (Math/pow (+ 1 factor) 3)
                 letter))
    ;; (when-let [glooby (first (keep :glooby
    ;;                                (lib/entities s)))])
    ))

;;
;; 1. high excitability: explore
;; 2. low excitability until blerps resolve 1:1 with
;;    conceptrinos, conceptrons
;;    (concept elements, mesoscale ideas)
;;    (~ slipnet level)
;; 3. High excitability again but now with the concept level
;;    restricting
;;
(defn blerp-idea-pump
  [{:keys [previous-excitability
           phase]}]
  ;; phase:
  ;;
  ;; explore:
  ;; increase excitability
  ;; - .. go into narrow?
  ;; - .. stop explore?
  ;; - .. osc
  ;;
  ;;
  ;; narrow:
  ;;  blerp resolve case:;
  ;;  Idea1:
  ;;   - all blerp particles completely on a single
  ;;     letter of the alphabet
  ;;   - update conceptrons
  ;;
  ;;  blerp not resolved:
  ;;   - decrease excitability
  ;;
  ;;  blerps are allowed to be *gone*
  ;;
  ;;
  )

(defn update-glue-space [])







;; ------------------------------------------
;; glooby
;;
;; glooby
;; glues   ordinary objects
;;
;; glueby
;;
;; global glue object builder y
;;
;; goggers
;;
;;

;; 1. for each blerp
;; 2. count overlaps with the alphabet ?
;;
;;


;;
;; blerp:
;;
;; brownian local explorer resonator particle
;;

;;
;;
;; ----------------------------
;; blerp layer
;;
;;
;; particle fields
;;
;;
;; ----------------------------
;; conceptronic layer
;;
;;
;;


;; -----------
;; Jessica Flack, "Architecture of collective computation":
;;
;;
;; W - the percievable state of the world
;; 'exogonous ground truths', social variables
;;
;; X - nodes on the microscopic circuit
;;     computed from the W's
;;
;; Y - macroscopic coarse grainings
;;
;; Z - outputs made possible by Y
;;

;; ---
;; W - 'world'
;; X - 'blerp layer'
;; Y - (Mitchels + Hofstaders slipnet)
;; Z -
;;

(defn env [_state] {})

(defn draw-state
  [state]
  ;; (q/background 255)
  ;; (q/with-translation [0 0] (wave/draw))
  ;; (q/color-mode :rbg)
  ;; (q/with-translation [0 500] (wave/draw))

  ;; (if (< 1 (q/random 2)) (q/color-mode :hsb))
  (q/color-mode :hsb)
  ;; (q/fill 0 0 0 50)
  ;; (q/fill 255 255 255 50)
  ;; (if (< 1 (q/random 2))
  ;;   (q/color-mode :hsb))
  (q/fill
   (lib/with-alpha
     (lib/->hsb (defs/color-map
                  (rand-nth
                   [
                    ;; :cyan
                    :black
                    ;; :white
                    ]
                   ))
                ;; (val
                ;;  (rand-nth
                ;;   (into [] defs/color-map)))
                )
     0.1))
  (q/rect 0 0 (* 2 2560) (* 2 1920))
  ;; (if (even? (mod (q/seconds) 2))
  ;;   (q/color-mode :hsb)
  ;;   (q/color-mode :rgb))
  (q/color-mode :hsb)
  (do (q/stroke-weight 0) (lib/draw-entities state))
  ;; (q/with-translation [0 -200] (wave/draw))
  )

(defn update-entity
  [entity state env]
  (-> entity
      (lib/update-body state)
      lib/brownian-motion
      lib/friction
      lib/dart-distants-to-middle
      lib/move-dragged
      lib/update-rotation
      lib/update-position
      (lib/update-sensors env)
      lib/activation-decay
      lib/activation-shine
      lib/shine
      lib/update-lifetime))

(defn update-state
  [state]
  (let [env (lib/env state)
        current-tick (q/millis)
        _ (def speed (:time-speed (lib/controls)))
        dt (* (:time-speed (lib/controls))
              (/ (- current-tick (:last-tick state 0)) 1000))]
    (binding [*dt* dt]
      (->
        state
        (update :t (fnil inc 0))
        (assoc :last-tick current-tick)
        ;; -----------------
        lib/kill-entities
        ;; -------------------
        lib/apply-update-events
        lib/update-update-functions
        lib/update-state-update-functions
        (lib/update-ents-parallel
          #(update-entity % state env))
        lib/update-late-update-map
        lib/transduce-signals
        ;; those 2 are heavy,
        lib/track-components
        lib/track-conn-lines
        ;; also heavy:
        lib/update-collisions
        ;;
        ;; phy/physics-update-2d
        lib/update-timers-v2))))


(defn setup
  [controls]
  (q/frame-rate 60)
  (q/rect-mode :center)
  (q/color-mode :hsb)
  (q/text-size defs/glyph-size)
  (q/text-font (q/create-font "Fira Code Bold"
                              defs/glyph-size)
               defs/glyph-size)
  (let [state {:controls controls :on-update []}
        state (-> state
                  lib/setup-version)]
    state))

(defn sketch
  [{:as controls
    :keys [width height sketch-id]
    :or {height 800 width 1000}}]
  (q/sketch
   :events-q {:sketch-id sketch-id}
   :size
   ;; hard coding my monitor, else it was going to
   ;; another monitor
   [1920 1080]
   ;; [600 600]
   ;; [800 800]
   :setup (comp (:setup controls identity)
                (partial setup controls))
   :update #'update-state
   :draw #'draw-state
   ;; :features [:keep-on-top]
   :middleware [m/fun-mode m/navigation-2d
                picture-jitter/picture-jitter
                evtq/events-middleware
                time-warp/time-warp]
   :navigation-2d {:modifiers {:mouse-dragged #{:shift}
                               :mouse-wheel #{:shift}}}
   :title "hyper-substrates"
   :key-released (fn [state event] state)
   :mouse-pressed (fn [s e]
                    (if (and (q/key-pressed?)
                             (= (q/key-modifiers)
                                #{:shift}))
                      s
                      (lib/mouse-pressed s e)))
   :mouse-dragged (fn [s e] s)
   :mouse-released (fn [s e]
                     (if (and (q/key-pressed?)
                              (= (q/key-modifiers)
                                 #{:shift}))
                       s
                       (lib/mouse-released s e)))
   :mouse-wheel (fn [s e]
                  (if (and (q/key-pressed?)
                           (= (q/key-modifiers) #{:shift}))
                    s
                    (lib/mouse-wheel s e)))))



(defn draw-grid
  [{:as e :keys [grid-width spacing elements draw-element]}]
  (let [[x y] (lib/position e)]
    (doall (for [i (range (count elements))
                 :let [coll (mod i grid-width)
                       row (quot i grid-width)]]
             (let [x (+ x (* coll spacing))
                   y (+ y (* row spacing))]
               (q/with-translation
                   [x y]
                 (draw-element e (elements i))))))
    (doall (for [i (range (count elements))
                 :let [coll (mod i grid-width)
                       row (quot i grid-width)]]
             (let [x (+ x (* coll spacing))
                   y (+ y (* row spacing))]
               (q/with-translation
                   [x y]
                 (draw-element e (elements i))))))))


(defn draw-grid-2
  [{:as e :keys [spacing]} elements]
  (q/with-translation
    (lib/position e)
    (doseq [[i row] (map-indexed vector elements)]
      (doseq [[j elm] (map-indexed vector row)]
        (let [elm (py.. elm item)]
          (let [x (* i spacing)
                y (* j spacing)]
            (q/with-translation
              [x y]
              ((:draw-element e) e elm))))))))

(defn field-map
  [state]
  (into {}
        (comp (filter :particle-id)
              (map (juxt :particle-id :particle-field)))
        (lib/entities state)))

(defn blerp-retina
  [{:as opts
    :keys [pos spacing grid-width color particle-id]}]
  (->
    (lib/->entity
      :q-grid
      (merge
        opts
        {:draw-element
         (fn [_ elm]
           (when-not (zero? elm)
             (q/with-stroke
                 (when (even? (q/millis)) defs/white)
                 (q/with-fill (lib/->hsb color)
                   (q/ellipse 0 0 8 8)))))
         :draw-functions
         {:grid (fn [e]
                  (draw-grid-2
                   e
                   (p/read-particles
                    (:particle-field e))))}
         :elements []
         :grid-width grid-width
         :particle-field
         (merge
          (assoc (p/grid-field
                  grid-width
                  [p/attenuation-update
                   p/vacuum-babble-update p/decay-update
                   ;; (partial p/pull-update
                   ;; :south)
                   p/brownian-update
                   p/reset-weights-update
                   ;; p/reset-excitability
                   p/reset-excitability-update])
                 :size grid-width
                 :activations
                 (pyutils/ensure-torch
                  (dtt/->tensor
                   (repeatedly
                    (* grid-width grid-width)
                    #(if (< (rand) 0.05) 1.0 0.0))
                   :datatype :float32)))
          (select-keys opts
                       [:decay-factor :attenuation-factor
                        :vacuum-babble-factor]))
         :particle-id particle-id
         :spacing spacing
         :transform (lib/->transform pos 0 0 1)}))
    #_(lib/live [:blerp-resonate #'update-blerp])
    (lib/live [:particle
               (lib/every-n-seconds
                 0.2
                 (fn [e s _]
                   (update e
                           :particle-field
                           p/update-grid-field)))])))


(defn attenuation-ball
  []
  (lib/->entity
   :circle
   {:color (defs/color-map :cyan)
    :kinetic-energy 0.5
    :particle? true
    :draggable? true
    :id :attenuation-ball
    :transform (lib/->transform
                (lib/rand-on-canvas-gauss 0.2)
                30
                30
                1)}))

(defn rubber-ball-values
  [e]
  (let [[x y] (lib/position e)]
    [(/ (+ x 100) (- (q/width) 100))
     (/ (+ y 100) (- (q/height) 100))]))

(defmethod lib/setup-version :berp-retina-f
  [state]
  (->
    state
    (lib/append-ents
      [(attenuation-ball)
       ;; (blerp-retina {:color (:orange
       ;; defs/color-map)
       ;;                :grid-width grid-width
       ;;                :interactions [[:attracted
       ;;                                :heliotrope]]
       ;;                :particle-id :orange
       ;;                :pos [50 50]
       ;;                :spacing 20})
       ;; (blerp-retina {:color (:heliotrope
       ;; defs/color-map)
       ;;                :grid-width grid-width
       ;;                :interactions [[:attracted
       ;;                :orange]]
       ;;                :particle-id :heliotrope
       ;;                :pos [50 50]
       ;;                :spacing 20})
       (-> (blerp-retina
             {:attenuation-factor (/ 1 2)
              :color (:green-yellow defs/color-map)
              :decay-factor (/ 10 100)
              :grid-width grid-width
              :interactions [[:attracted :orange]]
              :particle-id :green-yellow
              :pos [200 200]
              :spacing 20
              :vacuum-babble-factor (/ 1 100)})
           (lib/live
             (lib/every-n-seconds
               0.1
               (fn [e s k]
                 (let [[a b] (rubber-ball-values
                               ((lib/entities-by-id s)
                                 :attenuation-ball))]
                   (-> e
                       (assoc-in [:particle-field
                                  :vacuum-babble-factor]
                                 (* a 0.02))
                       (assoc-in [:particle-field
                                  :attenuation-factor]
                                 b)))))))])))


(sketch
 {:background-color 0
  :time-speed 3
  :v :berp-retina-f})

(comment
  (do (reset! lib/the-state nil) (System/gc)))
