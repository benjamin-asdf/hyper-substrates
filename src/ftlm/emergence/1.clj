(ns ftlm.emergence.1
  (:require
   [animismic.lib.particles :as p]
   [fastmath.core :as fm]
   [clojure.java.io :as io]
   [clojure.string :as str]
   [clojure.data.json :as json]
   [quil.middleware :as m]
   [ftlm.vehicles.art.lib :refer [*dt*] :as lib]
   [ftlm.vehicles.art.defs :as defsc]
   [ftlm.vehicles.art.extended :as elib]
   [tech.v3.datatype.unary-pred :as unary-pred]
   [tech.v3.datatype.functional :as f]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [quil.core :as q]
   [ftlm.vehicles.art.defs :as defs]
   [tech.v3.datatype.functional :as f]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [tech.v3.datatype.bitmap :as bitmap]
   [fastmath.random :as fm.rand]
   ;; [bennischwerdtner.hd.binary-sparse-segmented :as
   ;; hd]
   [bennischwerdtner.hd.core :as hd]
   [bennischwerdtner.pyutils :as pyutils]
   [tech.v3.datatype.unary-pred :as unary-pred]
   [tech.v3.datatype.argops :as dtype-argops]
   [bennischwerdtner.sdm.sdm :as sdm]
   [bennischwerdtner.hd.item-memory :as item-memory]
   [bennischwerdtner.hd.impl.item-memory-torch :as
    item-memory-torch]
   [bennischwerdtner.hd.codebook-item-memory :as codebook]
   [bennischwerdtner.hd.ui.audio :as audio]
   [bennischwerdtner.hd.data-next :as hdd]
   [ftlm.vehicles.art.physics :as phy]
   [animismic.lib.blerp :as b]
   [animismic.lib.particles-core :as pe]
   [ftlm.vehicles.cart :as cart]
   [animismic.lib.vehicles :as v]
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :refer [py. py..] :as py]
   [ftlm.disco.middlewares.picture-jitter :as
    picture-jitter]))

(def ant-types
  {:caretaker {:audio-freq 300
               :color (:navajo-white defs/color-map)
               :exploration-drive 0.2
               :explore-around-yellow-heart 0
               :hunger-for-green-food 0
               :kind :caretaker
               :love-for-yellow-heart 10}
   :gatherer {:audio-freq 350
              :color "#adff2f"
              :exploration-drive 3
              :explore-around-yellow-heart 0
              :hunger-for-green-food 6
              :kind :gatherer
              :love-for-yellow-heart 1}
   :soldier {:audio-freq 500
             :color (:red defs/color-map)
             :exploration-drive 2
             :explore-around-yellow-heart 2
             :hunger-for-green-food 2
             :kind :soldier
             :love-for-yellow-heart 1}
   :worker {:audio-freq 400
            :color "#00bfff"
            :exploration-drive 1
            :explore-around-yellow-heart 4
            :hunger-for-green-food 0
            :kind :worker
            :love-for-yellow-heart 0}})

(defn the-sound-of-birth
  []
  [(audio/->audio {:duration 0.1 :frequency 300})
   (audio/->audio {:duration 0.3 :frequency 350})])

(def glyph-size 18)

(defn vehicle-death
  [s e]
  (future
    (audio/play!
     (audio/->audio {:duration 0.1
                     :frequency 600}))
    (audio/play!
     (audio/->audio
      {:duration 0.1
       :frequency 150})))
  (let [new-e (-> e
                  (assoc :lifetime 0.2)
                  (lib/live (lib/->grow 0.1)))]
    (-> (lib/+explosion s e)
        (assoc-in [:eid->entity (:id e)] new-e))))

(defn draw-state
  [state]
  (q/background (lib/->hsb (-> state
                               :controls
                               :background-color)))
  (q/stroke-weight 0)
  (lib/draw-entities state))

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

(defn update-state-inner
  [state dt current-tick]
  (let [env (lib/env state)
        new-state (binding [*dt* dt]
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
                     lib/apply-events
                     (lib/update-ents-parallel
                      #(update-entity % state env))
                     lib/transduce-signals
                     ;; those 2 are heavy,
                     lib/track-components
                     lib/track-conn-lines
                     ;; also heavy:
                     lib/update-collisions
                     ;;
                     phy/physics-update-2d
                     lib/update-late-update-map
                     ;; -----------------
                     ))]
    (merge state new-state)))

(defn update-state
  [state]
  (let [current-tick (q/millis)
        ;; state (update state
        ;;               :controls
        ;;               merge
        ;;               (user-controls/controls))
        dt (*
            (:time-speed (lib/controls))
            (/ (- current-tick
                  (:last-tick @lib/the-state 0))
               1000.0))]
    (lib/update-timers dt)
    (merge state (swap! lib/the-state update-state-inner dt current-tick))))

(defn setup
  [controls]
  (q/frame-rate 60)
  (q/rect-mode :center)
  (q/color-mode :hsb)
  (q/text-size glyph-size)
  (q/text-font (q/create-font "Fira Code Bold" glyph-size) glyph-size)

  ;; (q/background (lib/->hsb (-> controls
  ;;                              :background-color)))
  (let [state {:controls controls :on-update []}
        state (-> state lib/setup-version)]
    (reset! lib/the-state state)))

(defn sketch
  [{:as controls
    :keys [width height]
    :or {height 800 width 1000}}]
  (q/sketch
   :size
   ;; hard coding my monitor, else it was going to
   ;; another monitor
   ;; [2560 1920]
   ;; [1800 1200]
   [1200 900]
   ;; [500 500]
   :setup (partial setup controls)
   :update #'update-state
   :draw #'draw-state
   ;; :features [:keep-on-top]
   :middleware [m/fun-mode
                ;; (fn [opts])
                m/navigation-2d
                picture-jitter/picture-jitter]
   :navigation-2d {:modifiers {:mouse-dragged #{:shift}
                               :mouse-wheel #{:shift}}}
   :title "hyper-substrates"
   :key-released (fn [state event] state)
   :mouse-pressed (comp #(reset! lib/the-state %)
                        (fn [s e]
                          (if (and (q/key-pressed?)
                                   (= (q/key-modifiers)
                                      #{:shift}))
                            s
                            (lib/mouse-pressed s e))))
   :mouse-dragged (comp #(reset! lib/the-state %)
                        (fn [s e] s))
   :mouse-released (comp #(reset! lib/the-state %)
                         (fn [s e]
                           (if (and (q/key-pressed?)
                                    (= (q/key-modifiers)
                                       #{:shift}))
                             s
                             (lib/mouse-released s e))))
   :mouse-wheel (comp #(reset! lib/the-state %)
                      (fn [s e]
                        (if (and (q/key-pressed?)
                                 (= (q/key-modifiers)
                                    #{:shift}))
                          s
                          (lib/mouse-wheel s e))))
   :on-close (reset! lib/the-state nil)))

(defn update-intensity-osc
  [e s _]
  (let [speed 1
        cycle-duration 20000
        e (update
            e
            :intensity-factor
            (fn [x]
              (let [fade-factor (-> (* (/ (q/millis)
                                          cycle-duration)
                                       q/TWO-PI)
                                    (Math/sin)
                                    (Math/abs))
                    wave-value (* fade-factor
                                  (+ x (* lib/*dt* speed)))]
                wave-value)))]
    (assoc e
           :intensity (+ 15 (* 10 (:intensity-factor e))))))

(defn ->ray-source
  ([] (->ray-source (lib/rand-on-canvas-gauss 0.2)))
  ([pos]
   (let [[e]
         (lib/->ray-source
          {:color (:mint defs/color-map)
           :intensity 50
           :intensity-factor 1
           :kinetic-energy 1
           :on-collide-map
           {:burst (lib/cooldown 5 lib/burst)}
           :on-double-click-map
           {:orient-towards-me
            (fn [e s k]
              {:updated-state
               (lib/update-ents
                s
                (fn [ent]
                  (lib/orient-towards
                   ent
                   (lib/position e))))})}
           :on-drag-start-map
           {:survive (fn [e s k]
                       (dissoc e :lifetime))}
           :particle? true
           :pos pos
           :ray-kind :yellow-heart
           :scale 1})]
     (->
      e
      #_(lib/live
         [:rays
          (lib/every-n-seconds
           1.5
           (fn [e s _]
             {:updated-state
              (let [bodies (into []
                                 (filter :body?
                                         (lib/entities s)))
                    a (first (shuffle bodies))
                    b e]
                (when (and a b)
                  (->
                   s
                   (lib/append-ents
                    [(merge
                      (lib/->connection-bezier-line
                       a
                       b)
                      {:lifetime (lib/normal-distr
                                  2
                                  1)})]))))}))])
      #_(lib/live [:colors
                   (lib/every-n-seconds
                    0.5
                    (fn [e s k]
                      (assoc e
                             :color (defs/color-map
                                      (rand-nth
                                       [:green-yellow
                                        :heliotrope])))))])
      #_(lib/live [:kinetic
                   (lib/every-n-seconds
                    5
                    (fn [e s k]
                      (assoc e
                             :kinetic-energy
                             (lib/normal-distr 5 2))))])
      (lib/live
       [:circular-shine-radio
        (lib/every-n-seconds
         (fn [] (lib/normal-distr (/ 1.5 3) (/ 1.5 3)))
         (fn [ray s k]
           {:updated-state
            (lib/append-ents
             s
             [(let [e (lib/->circular-shine-1 ray)]
                (->
                 e
                 (assoc :color (lib/with-alpha
                                 (:yellow
                                  defs/color-map)
                                 0))
                 (assoc :stroke-weight 3)
                 (assoc :stroke (:color ray))
                 (assoc :on-update
                        [(lib/->grow
                          (* 2
                             (+ 1
                                (:intensity-factor
                                 ray
                                 0))))])
                 (assoc :lifetime (lib/normal-distr
                                   3
                                   (Math/sqrt
                                    3)))))])}))])
      #_(lib/live
         [:circular-shine-field
          (lib/every-n-seconds
           (fn [] (lib/normal-distr (/ 1.5 3) (/ 1.5 3)))
           (fn [ray s k]
             {:updated-state
              (lib/append-ents
               s
               [(let [e (lib/->circular-shine-1 ray)]
                  (-> e
                      (assoc :color
                             ((rand-nth
                               [:black :hit-pink
                                :cyan :heliotrope])
                              defs/color-map))
                      ;; (assoc :stroke-weight
                      ;; 3)
                      ;; (assoc :stroke (:color
                      ;; ray))
                      (assoc
                       :on-update
                       [(lib/->grow
                         (* 2
                            (+ 1
                               (:intensity-factor
                                ray
                                0))))])
                      (assoc :lifetime
                             (lib/normal-distr
                              10
                              (Math/sqrt
                               5)))))])}))])
      (lib/live [:intensity-osc update-intensity-osc])))))

(defn food-particle
  []
  (let [poison?
        ;; false
        (zero? (fm.rand/flip 0.99))
        ]
    (lib/->entity
      :circle
      {
       :hidden? true
       :collides? true
       :draggable? true
       :kinetic-energy 0.5
       :particle? true
       :poison? poison?
       :transform (lib/->transform
                    (lib/rand-position-around
                      (let [dist 150]
                        (rand-nth
                          [[dist dist]
                           [(lib/from-right dist)
                            (lib/from-bottom dist)]
                           [dist (lib/from-bottom dist)]
                           [(lib/from-right dist) dist]]))
                      80)
                    80
                    80
                    1)
       :on-collide-map
         {:eaten
            (fn [e other state k]
              (if (:lifetime e)
                e
                (if-not (:ant-data other)
                  e
                  (let
                    [new-e
                       (->
                         e
                         (assoc :acceleration 300)
                         (assoc :kinetic-energy 0.2)
                         ;; (assoc :particle? false)
                         (lib/orient-towards (lib/position
                                               other))
                         (assoc-in
                           [:on-update-map :grow]
                           (lib/every-now-and-then
                             0.5
                             (fn [e s k]
                               (update-in
                                 e
                                 [:transform :scale]
                                 *
                                 (+
                                   1
                                   (*
                                     0.002
                                     (q/random-gaussian)))))))
                         (assoc :lifetime 2))]
                    {:updated-state
                       (cond->
                           state
                           poison?
                           (vehicle-death other)
                           :always
                           (assoc-in [:eid->entity (:id e)]
                                     new-e))}))))}
       :components
         [(lib/->ray-source-1
            {:anchor :middle-middle
             :clickable? false
             :color ((if poison? :fruit-salad :mint)
                      defs/color-map)
             :draggable? false
             :food? true
             :intensity 1
             :ray-kind :green-food
             :shinyness false
             :transform (lib/->transform [0 0] 15 15 1)})
          (lib/->odor-source
           {:anchor :middle-middle
            :decay-rate 2
            :fragrances #{:food}
            :intensity 40})]})))

;; (defn overlay-text
;;   []
;;   (into
;;     [(lib/->entity :overlay-rect
;;                    {:draw-functions
;;                       {:f (fn [e]
;;                             (q/push-style)
;;                             (q/rect-mode :corner)
;;                             (q/fill (lib/with-alpha
;;                                       (lib/->hsb defs/black)
;;                                       0.5))
;;                             (q/rect (lib/from-right 220)
;;                                     (+ (/ (q/height) 2) 250)
;;                                     200
;;                                     200)
;;                             (q/pop-style))}})]
;;     (map-indexed
;;       (fn [idx ant-kind]
;;         (-> (elib/->text
;;               {:color defs/white
;;                :text (str idx)
;;                :text-idx idx
;;                :transform (lib/->transform
;;                             [(lib/from-right 200)
;;                              (+ (/ (q/height) 2)
;;                                 300
;;                                 (* (+ glyph-size 10) idx))]
;;                             50
;;                             50
;;                             1)})
;;             (lib/live
;;               [:update-text
;;                (fn [e s k]
;;                  (-> e
;;                      (assoc :text


;;                             (str
;;                              (name ant-kind)
;;                              ": "
;;                              (ant-kind
;;                               (frequencies
;;                                (map :kind
;;                                     (keep
;;                                      :ant-data
;;                                      (lib/entities
;;                                       s)))))))))])))
;;       (keys ant-types))))


(defn overlay-text
  []
  [(lib/->entity :overlay-rect
                 {:draw-functions
                  {:f (fn [e]
                        (q/push-style)
                        (q/rect-mode :corner)
                        (q/fill (lib/with-alpha
                                  (lib/->hsb defs/black)
                                  0.8))
                        (q/rect (lib/from-right 260)
                                (+ (/ (q/height) 2) 250)
                                300
                                150)
                        (q/pop-style))}})
   (-> (elib/->text {:color defs/white
                     :text ""
                     :transform
                     (lib/->transform
                      [(lib/from-right 250)
                       (+ (/ (q/height) 2) 250)]
                      50
                      50
                      1)})
       (lib/live
        [:update-text
         (fn [e s k]
           (let [ants-info (reverse (sort-by
                                     val
                                     (frequencies
                                      (map :kind
                                           (keep
                                            :ant-data
                                            (lib/entities
                                             s))))))]
             (-> e
                 (assoc
                  :text
                  (with-out-str
                    (clojure.pprint/print-table
                     (for [[k v] ants-info]
                       {:name k
                        :freq v})))))))]))])

(defn draw-grid
  [{:as e :keys [grid-width spacing elements]}]
  (let [draw-element (fn [elm]
                       (q/with-fill (lib/->hsb-vec (:color elm))
                         (q/rect 0 0 20 20 5)))]
    (let [[x y] (lib/position e)]
      (doall (for [i (range (count elements))
                   :let [coll (mod i grid-width)
                         row (quot i grid-width)]]
               (let [x (+ x (* coll spacing))
                     y (+ y (* row spacing))]
                 (q/with-translation [x y]
                                     (draw-element
                                      (elements i)))))))))

(defn smell-history-overview
  []
  [(->
     (lib/->entity
       :smell-history
       {:draw-functions
          {:f (fn [e]
                (q/push-style)
                (q/rect-mode :corner)
                (q/with-translation
                  (lib/position e)
                  (q/fill (lib/with-alpha (lib/->hsb-vec
                                            defs/white)
                                          0.1))
                  (q/rect
                    0
                    0
                    (+
                     (* 25
                        (or (count (:smell-history-to-show
                                    e))
                            0))
                     (if (zero? (count (:smell-history-to-show e)))
                       0
                       15
                       ))
                    40))
                (q/pop-style)
                (q/with-translation
                  [20 20]
                  (draw-grid
                    (merge e
                           {:elements
                              (into
                                []
                                (map (fn [p]
                                       {:color (-> ant-types
                                                   p
                                                   :color)})
                                  (:smell-history-to-show
                                    e)))
                            :grid-width 100
                            :spacing 25}))))}
        :transform (lib/->transform [150 60] 200 50 1)})
     (lib/live (fn [e s k]
                 (assoc e
                   :smell-history-to-show
                     (-> s
                         :eid->entity
                         (get (:id (:selection s)))
                         :smell-history)))))])


(defn add-food-particle
  [state]
  (lib/append-ents state [(food-particle)]))



;;
;;
;; [=== ]    soldier
;; [=======  ]  worker
;; [============ ]
;;
;; smell +1
;; decays
;;
;; every now and then... <- random
;; if they are not balanced,
;; swap to the least
;;

;; (defn love-green-food
;;   [intensity]
;;   [[:brain/neuron :green-food-love-arousal
;;     {:arousal intensity
;;      :arousal-neuron? true
;;      :on-update [(fn [e _]
;;                    (update-in e
;;                               [:activation]
;;                               (fnil + 0)
;;                               (* 0.1 (:arousal e))))]}]
;;    ;; ------------------------------------
;;    [:cart/sensor :green-food-ray-sensor
;;     {:anchor :top-right
;;      :modality :rays
;;      :ray-kind :green-food}]
;;    [:cart/sensor :green-food-sensor-left
;;     {:anchor :top-left
;;      :modality :rays
;;      :ray-kind :green-food}]
;;    ;; ---------------------------
;;    [:brain/connection :green-food-arousal-right
;;     {:destination [:ref :motor-bottom-right]
;;      :f :excite
;;      :hidden? true
;;      :on-update-map {:gain (lib/every-n-seconds
;;                              (fn [] (lib/normal-distr 1 1))
;;                              (fn [e s k]
;;                                (assoc-in e
;;                                  [:transduction-model :gain]
;;                                  (rand))))}
;;      :source [:ref :green-food-love-arousal]}]
;;    [:brain/connection :green-food-arousal-left
;;     {:destination [:ref :motor-bottom-left]
;;      :f :excite
;;      :hidden? true
;;      :on-update-map {:gain (lib/every-n-seconds
;;                              (fn [] (lib/normal-distr 1 1))
;;                              (fn [e s k]
;;                                (assoc-in e
;;                                  [:transduction-model :gain]
;;                                  (rand))))}
;;      :source [:ref :green-food-love-arousal]}]
;;    ;; --------------------------------
;;    [:brain/connection :green-food-connection-right
;;     {:destination [:ref :motor-bottom-right]
;;      :f (lib/->weighted (* -1 intensity))
;;      :hidden? true
;;      :source [:ref :green-food-ray-sensor]}]
;;    [:brain/connection :green-food-connection-left
;;     {:destination [:ref :motor-bottom-left]
;;      :f (lib/->weighted (* -1 intensity))
;;      :hidden? true
;;      :source [:ref :green-food-sensor-left]}]])


;;
;; hungry
;;   seeking
;;   feeding
;;

(defn hunger-for-green-food
  [intensity]
  [;; [:cart/sensor :green-food-sensor-left
   ;;  {:anchor :top-left
   ;;   :modality :rays
   ;;   :ray-kind :green-food}]
   ;; [:cart/sensor :green-food-sensor-right
   ;;  {:anchor :top-right
   ;;   :modality :rays
   ;;   :ray-kind :green-food}]
   [:cart/sensor :green-food-sensor-left
    {:anchor :top-left :fragrance :food :modality :smell}]
   [:cart/sensor :green-food-sensor-right
    {:anchor :top-right :fragrance :food :modality :smell}]
   ;; --------------------------------
   [:brain/connection :green-food-connection-a
    {:destination [:ref :motor-bottom-left]
     :f (lib/->weighted -1)
     :hidden? true
     :on-update-map {:gain (fn [e s k]
                             (assoc-in e
                               [:transduction-model :gain]
                               (* 0.3 (intensity))))}
     :source [:ref :green-food-sensor-left]}]
   [:brain/connection :green-food-connection-b
    {:destination [:ref :motor-bottom-right]
     :f (lib/->weighted -1)
     :hidden? true
     :on-update-map {:gain (fn [e s k]
                             (assoc-in e
                               [:transduction-model :gain]
                               (* 0.3 (intensity))))}
     :source [:ref :green-food-sensor-right]}]])


(defn exploration-wires
  [intensity]
  [[:brain/neuron :exploration-arousal
    {:arousal (intensity)
     :arousal-neuron? true
     :on-update [(fn [e _]
                   (update-in e
                              [:activation]
                              (fnil + 0)
                              (intensity)))]}]
   [:brain/connection :green-food-arousal-right
    {:destination [:ref :motor-bottom-right]
     :f :excite
     :hidden? true
     :on-update-map
     {:gain (lib/every-n-seconds
             0.2
             (fn [e s k]
               (assoc-in e
                         [:transduction-model :gain]
                         (+ 0.5 (lib/normal-distr 0.1 0.1)))))}
     :source [:ref :exploration-arousal]}]
   [:brain/connection :green-food-arousal-left
    {:destination [:ref :motor-bottom-left]
     :f :excite
     :hidden? true
     :on-update-map
     {:gain (lib/every-n-seconds
             0.2
             (fn [e s k]
               (assoc-in e
                         [:transduction-model :gain]
                         (+ 0.5 (lib/normal-distr 0.1 0.1)))))}
     :source [:ref :exploration-arousal]}]])

(defn love-for-yellow-heart
  [intensity]
  (let [seeking? (atom true)]
    [[:brain/neuron :yellow-anger-arousal
      {:arousal (intensity)
       :arousal-neuron? true
       :on-update
       [(fn [e _]
          (update-in e
                     [:activation]
                     (fnil + 0)
                     (*
                      ;; (if @seeking? 0.2
                      ;; 0.1)
                      0.2
                      (intensity))))]}]
     ;; ------------------------------------
     [:cart/sensor :yellow-ray-sensor-left
      {:anchor :top-left
       :modality :rays
       :on-update-map {:flip-feed
                       (fn [e s k]
                         (cond (< 5 (:activation e 0))
                               (reset! seeking? false)
                               (< (:activation e 0) 1)
                               (reset! seeking? true)
                               :else nil)
                         e)}
       :ray-kind :yellow-heart}]
     [:cart/sensor :yellow-ray-sensor-right
      {:anchor :top-right
       :modality :rays
       :ray-kind :yellow-heart}]
     ;; ---------------------------
     [:brain/connection :yellow-arousal-right
      {:destination [:ref :motor-bottom-right]
       :f :excite
       :hidden? true
       :on-update-map
       {:gain (lib/every-n-seconds
               0.2
               (fn [e s k]
                 (assoc-in e
                           [:transduction-model :gain]
                           (+ 0.5 (lib/normal-distr 0.1 0.1)))))}
       :source [:ref :yellow-anger-arousal]}]
     ;; ----------------------
     [:brain/connection :yellow-arousal-left
      {:destination [:ref :motor-bottom-left]
       :f :excite
       :hidden? true
       :on-update-map
       {:gain (lib/every-n-seconds
               0.2
               (fn [e s k]
                 (assoc-in e
                           [:transduction-model :gain]
                           (+ 0.5 (lib/normal-distr 0.1 0.1)))))}
       :source [:ref :yellow-anger-arousal]}]
     ;; --------------------------------
     [:brain/connection :yellow-connection-a
      {:destination [:ref :motor-bottom-right]
       :f (lib/->weighted -1)
       :hidden? true
       :on-update-map {:gain (fn [e s k]
                               (assoc-in e
                                         [:transduction-model :gain]
                                         (* 0.3 (intensity))
                                         ))}
       :source [:ref :yellow-ray-sensor-right]}]
     [:brain/connection :yellow-connection-b
      {:destination [:ref :motor-bottom-left]
       :f (lib/->weighted -1)
       :hidden? true
       :on-update-map
       {:gain
        (fn [e s k]
          (assoc-in e
                    [:transduction-model :gain]
                    (* 0.3 (intensity))))}
       :source [:ref :yellow-ray-sensor-left]}]]))

(defn explore-around-yellow-heart
  [intensity]
  (let [seeking? (atom true)]
    [[:brain/neuron :yellow-explore-anger-arousal
      {:arousal (intensity)
       :arousal-neuron? true
       :on-update
       [(fn [e _]
          (update-in e
                     [:activation]
                     (fnil + 0)
                     (*
                      0.3
                      (intensity))))]}]
     ;; ------------------------------------
     [:cart/sensor :yellow-explore-ray-sensor-left
      {:anchor :top-left
       :modality :rays
       :ray-kind :yellow-heart}]
     [:cart/sensor :yellow-explore-ray-sensor-right
      {:anchor :top-right
       :modality :rays
       :ray-kind :yellow-heart}]
     ;; ---------------------------
     [:brain/connection :yellow-explore-arousal-right
      {:destination [:ref :motor-bottom-right]
       :f :excite
       :hidden? true
       :on-update-map
       {:gain (lib/every-n-seconds
               0.2
               (fn [e s k]
                 (assoc-in e
                           [:transduction-model :gain]
                           (+ 0.5 (lib/normal-distr 0.1 0.1)))))}
       :source [:ref :yellow-explore-anger-arousal]}]
     ;; ----------------------
     [:brain/connection :yellow-explore-arousal-left
      {:destination [:ref :motor-bottom-left]
       :f :excite
       :hidden? true
       :on-update-map
       {:gain (lib/every-n-seconds
               0.2
               (fn [e s k]
                 (assoc-in e
                           [:transduction-model :gain]
                           (+ 0.5 (lib/normal-distr 0.1 0.1)))))}
       :source [:ref :yellow-explore-anger-arousal]}]
     ;; --------------------------------
     [:brain/connection :yellow-explore-connection-a
      {:destination [:ref :motor-bottom-left]
       :f (lib/->weighted -1)
       :hidden? true
       :on-update-map {:gain (fn [e s k]
                               (assoc-in e
                                         [:transduction-model :gain]
                                         (* 0.5 (intensity))))}
       :source [:ref :yellow-explore-ray-sensor-right]}]
     [:brain/connection :yellow-explore-connection-b
      {:destination [:ref :motor-bottom-right]
       :f (lib/->weighted -1)
       :hidden? true
       :on-update-map
       {:gain (fn [e s k]
                (assoc-in e
                          [:transduction-model :gain]
                          (* 0.5 (intensity))))}
       :source [:ref :yellow-explore-ray-sensor-left]}]]))

;; -------------------------------

(defn vehicle-1
  ([] (vehicle-1 {}))
  ([body-opts]
   (let
     [[ant-type ant-data] (rand-nth (into [] ant-types))
      ;; ant-data (:caretaker ant-types)
      internal-state (atom ant-data)
      cart
        (cart/->cart
          {:body
             (merge
               {:ant-data ant-data
                :collides? true
                :on-collide-map
                  {:keep-track-of-who-i-meet
                     (lib/cooldown
                       0.5
                       (fn [e other state k]
                         (if (= (:id other)
                                (:last-guy-that-i-meet e))
                           e
                           (if-not (:ant-data other)
                             e
                             (do
                               #_(when (< (q/random 1) 0.05)
                                   (future
                                     (audio/play!
                                       (audio/->audio
                                         {:duration 0.2
                                          :frequency
                                            (rand-nth
                                              [600])}))))
                               {:updated-state
                                  (->
                                    state
                                    (assoc-in
                                      [:eid->entity (:id e)
                                       :last-guy-that-i-meet]
                                      (:id other))
                                    (update-in
                                      [:eid->entity (:id e)
                                       :smell-history]
                                      (fnil
                                        (fn [hist]
                                          (let
                                            [hist
                                               (conj
                                                 hist
                                                 (->
                                                   other
                                                   :ant-data
                                                   :kind))]
                                            (vec (take-last
                                                   10
                                                   hist))))
                                        []))
                                    (lib/append-ents
                                      [(let
                                         [sin
                                            (elib/sine-wave-machine
                                              100
                                              (lib/normal-distr
                                                500
                                                100))]
                                         (->
                                           (assoc
                                            (lib/->connection-line
                                             e
                                             other)
                                            :stroke-weight
                                            5
                                            :z-index
                                            -5
                                            :lifetime
                                            0.2
                                            ;; (-
                                            ;;  2
                                            ;;  (*
                                            ;;   0.01
                                            ;;   (+
                                            ;;    (:velocity
                                            ;;    e 1)
                                            ;;    (:velocity
                                            ;;    other
                                            ;;    1))))
                                            )
                                           (assoc-in
                                             [:on-late-update-map
                                              :flash]
                                             (let
                                               [base-color
                                                  (:color
                                                    e)]
                                               (fn [e s k]
                                                 (assoc e
                                                   :color
                                                     (lib/with-alpha
                                                       base-color
                                                       (sin))))))))]))}
                               ;; (-> e
                               ;;     (assoc
                               ;;     :last-guy-that-i-meet
                               ;;     (:id other))
                               ;;     ;;
                               ;;     (activation-flash
                               ;;     ;;  (:color e)
                               ;;     ;;  defs/white
                               ;;     ;;  (fn [ek]
                               ;;     ;;    (assoc ek
                               ;;     :color (:color
                               ;;     e))))
                               ;;     )
                             )))))}
                :on-double-click-map
                  {:die (fn [e s k]
                          {:updated-state
                             (vehicle-death s e)})}
                :on-update-map
                  {
                   ;; :workers-flip
                   ;; (lib/every-now-and-then
                   ;;  5
                   ;;  (fn [e s k]
                   ;;    (when
                   ;;        (= :worker (:kind (:ant-data e)))
                   ;;      (let [new-love (rand-nth [0 10])]
                   ;;        (swap! internal-state assoc :love-for-yellow-heart new-love)
                   ;;        (assoc
                   ;;         e
                   ;;         :color
                   ;;         ({0 (defs/color-map :worker-blue)
                   ;;           10 (defs/color-map :black)}
                   ;;          new-love))))))



                   :decides-to-flip
                   (lib/every-now-and-then
                    10
                    (fn [e s k]
                      ;; ant-flip
                      (if-not
                          (<= 5 (count (:smell-history e)))
                          e
                          (let [target-ant-type
                                (key
                                 (first
                                  (shuffle
                                   (first
                                    (partition-by
                                     val
                                     (sort-by
                                      val
                                      (merge
                                       (update-vals
                                        ant-types
                                        (constantly 0))
                                       (frequencies
                                        (:smell-history
                                         e)))))))))

                                ]
                            (if (= target-ant-type
                                   (-> e
                                       :ant-data
                                       :kind))
                              e
                              (do
                                (reset! internal-state
                                        (target-ant-type
                                         ant-types))
                                (let
                                    [new-e
                                     (->
                                      e
                                      (assoc
                                       :color
                                       (:color
                                        (target-ant-type
                                         ant-types)))
                                      (assoc
                                       :smell-history [])
                                      (assoc
                                       :ant-data
                                       (target-ant-type
                                        ant-types))
                                      (assoc
                                       :angular-velocity
                                       (rand-nth
                                        [-2 1 0 1
                                         2])))]
                                    {:updated-state
                                     (let
                                         [clockwise?
                                          (zero?
                                           (fm.rand/flip
                                            0.5))]
                                         (future
                                           (audio/play!
                                            (audio/->audio
                                             {:duration 0.2
                                              :frequency
                                              (:audio-freq
                                               (target-ant-type
                                                ant-types))})))
                                         (->
                                          s
                                          (assoc-in
                                           [:eid->entity
                                            (:id e)]
                                           new-e)
                                          (lib/append-ents
                                           (mapcat identity
                                                   (for
                                                       [p (repeatedly
                                                           5
                                                           (fn []
                                                             (let
                                                                 [pos
                                                                  (mapv
                                                                   +
                                                                   (lib/position
                                                                    e)
                                                                   (mapv
                                                                    #(*
                                                                      50
                                                                      %)
                                                                    (q/random-2d)))
                                                                  rotation
                                                                  ((if
                                                                       clockwise?
                                                                       +
                                                                       -)
                                                                   (lib/angle-between
                                                                    (lib/position
                                                                     e)
                                                                    pos)
                                                                   q/HALF-PI)]
                                                                 (lib/->entity
                                                                  :circle
                                                                  {:color
                                                                   (lib/with-alpha
                                                                     (lib/->hsb-vec
                                                                      defs/white)
                                                                     (abs
                                                                      (lib/normal-distr
                                                                       0.5
                                                                       1)))
                                                                   :lifetime
                                                                   0.8
                                                                   :acceleration
                                                                   1000
                                                                   ;; :angular-velocity
                                                                   ;; 20
                                                                   :transform
                                                                   (lib/->transform
                                                                    pos
                                                                    10
                                                                    10
                                                                    1 rotation)}))))]
                                                       [p
                                                        (assoc
                                                         (lib/->connection-line
                                                          p
                                                          e)
                                                         :z-index
                                                         -5
                                                         :stroke-weight
                                                         (abs
                                                          (lib/normal-distr
                                                           3
                                                           1)))
                                                        (lib/->entity
                                                         :circle
                                                         {:color
                                                          (lib/with-alpha
                                                            (lib/->hsb
                                                             defs/white)
                                                            0)
                                                          :lifetime
                                                          0.8
                                                          :on-update-map
                                                          {:grow
                                                           (fn
                                                             [e
                                                              s
                                                              k]
                                                             (update-in
                                                              e
                                                              [:transform
                                                               :scale]
                                                              +
                                                              (lib/normal-distr
                                                               1
                                                               1)))}
                                                          :stroke
                                                          (lib/with-alpha
                                                            (lib/->hsb-vec
                                                             defs/white)
                                                            (abs
                                                             (lib/normal-distr
                                                              0
                                                              0.2)))
                                                          :stroke-weight
                                                          (abs
                                                           (lib/normal-distr
                                                            20
                                                            10))
                                                          :transform
                                                          (let
                                                              [size
                                                               (abs
                                                                (lib/normal-distr
                                                                 5
                                                                 5))]
                                                              (lib/->transform
                                                               (lib/position
                                                                e)
                                                               size
                                                               size
                                                               1))})])))))})))))))}
                :scale 0.4
                :stroke-weight 0
                :vehicle-feel-color? true}
               (select-keys ant-data [:color])
               body-opts)
           :components
             (concat
               [[:cart/motor :motor-bottom-right
                 {:anchor :bottom-right
                  :corner-r 5
                  :hidden? true
                  :on-update [(lib/->cap-activation)]
                  :rotational-power 0.02}]
                [:cart/motor :motor-bottom-left
                 {:anchor :bottom-left
                  :corner-r 5
                  :hidden? true
                  :on-update [(lib/->cap-activation)]
                  :rotational-power 0.02}]
                ;; [
                ;;  ;; ---------------
                ;;  [:cart/sensor :sa
                ;;   {:anchor :top-right
                ;;   :modality :rays}]
                ;;  [:cart/sensor :sb
                ;;   {:anchor :top-left
                ;;   :modality :rays}]
                ;;  ;; ----------------
                ;;  ;; Temperature sensor
                ;;  [:cart/sensor
                ;;  :hot-temperature-sensor
                ;;   {:anchor :middle-middle
                ;;    :hot-or-cold :hot
                ;;    :modality
                ;;    :temperature}]
                ;; ----------------------------
                #_[:brain/neuron :arousal
                   {:arousal (lib/normal-distr 0.8 0.5)
                    :arousal-neuron? true
                    :on-update [(fn [e _]
                                  (update-in e
                                             [:activation]
                                             (fnil + 0)
                                             (* 0.4
                                                (:arousal
                                                  e))))]}]
                ;; ----------------------------
                #_[:brain/connection :_
                   {:destination [:ref :motor-bottom-right]
                    :f :excite
                    :hidden? true
                    :on-update-map
                      {:gain
                         (lib/every-n-seconds
                           (fn [] (lib/normal-distr 1 1))
                           (fn [e s k]
                             (assoc-in e
                               [:transduction-model :gain]
                               (rand))))}
                    :source [:ref :arousal]}]
                #_[:brain/connection :_
                   {:destination [:ref :motor-bottom-left]
                    :f :excite
                    :hidden? true
                    :on-update-map
                      {:gain
                         (lib/every-n-seconds
                           (fn [] (lib/normal-distr 1 1))
                           (fn [e s k]
                             (assoc-in e
                               [:transduction-model :gain]
                               (rand))))}
                    :source [:ref :arousal]}]
                ;;  ;; ----------------------------
                ;;  [:brain/connection :_
                ;;   {:decussates? false
                ;;    :destination [:ref
                ;;    :motor-bottom-right]
                ;;    :f (lib/->weighted -10)
                ;;    :hidden? true
                ;;    :source [:ref :sa]}]
                ;;  [:brain/connection :_
                ;;   {:decussates? false
                ;;    :destination [:ref
                ;;    :motor-bottom-left]
                ;;    :f (lib/->weighted -10)
                ;;    ;; :f :excite
                ;;    :hidden? true
                ;;    :source [:ref :sb]}]
                ;;  ;; ----------------------------
                ;;  [:brain/connection :_
                ;;   {:decussates? false
                ;;    :destination [:ref
                ;;    :motor-bottom-left]
                ;;    :f :excite
                ;;    :hidden? true
                ;;    :source [:ref
                ;;    :hot-temperature-sensor]}]
                ;;  [:brain/connection :_
                ;;   {:decussates? false
                ;;    :destination [:ref
                ;;    :motor-bottom-right]
                ;;    :f :excite
                ;;    :hidden? true
                ;;    :source [:ref
                ;;    :hot-temperature-sensor]}]
                ;;  ;; ----------------------------
                ;;  [:brain/connection :_
                ;;   {:decussates? true
                ;;    :destination [:ref
                ;;    :motor-bottom-right]
                ;;    :f :excite
                ;;    :hidden? true
                ;;    :on-update-map
                ;;    {:gain (fn [e s k]
                ;;             (assoc-in e
                ;;                       [:transduction-model
                ;;                       :gain]
                ;;                       #(*
                ;;                       2
                ;;                       %)))}
                ;;    :source [:ref :sb]}]
                ;;  [:brain/connection :_
                ;;   {:decussates? true
                ;;    :destination [:ref
                ;;    :motor-bottom-left]
                ;;    :f :excite
                ;;    :hidden? true
                ;;    :source [:ref :sa]}]]
                ;; ----------------------------
               ]
               (hunger-for-green-food
                 (fn []
                   (:hunger-for-green-food
                     @internal-state)))
               (love-for-yellow-heart
                 (fn []
                   (:love-for-yellow-heart
                     @internal-state)))
               (explore-around-yellow-heart
                 (fn []
                   (:explore-around-yellow-heart
                     @internal-state)))
               (exploration-wires (fn []
                                    (:exploration-drive
                                     @internal-state))))})]
     cart)))


(defn ->vehicle-field-1
  [grid-width]
  (merge
   (p/grid-field grid-width
                 [p/attenuation-update
                  p/vacuum-babble-update p/decay-update
                  p/brownian-update p/reset-weights-update
                  p/reset-excitability-update])
   {:activations (pyutils/ensure-torch
                  (dtt/->tensor
                   (repeatedly (* grid-width grid-width)
                               #(fm.rand/flip 0.5))
                   :datatype
                   :float32))
    :attenuation-factor 0
    :decay-factor (/ 1 10)
    :size grid-width
    :vacuum-babble-factor (/ 1 20)}))

(defn activation-flash
  [e base-color high-color kont]
  (let [sin (elib/sine-wave-machine 50 2000)
        end-timer (lib/set-timer 1)]
    (lib/live e
              [:flash
               (fn [e s k]
                 (if (lib/rang? end-timer)
                   (-> e
                       (update :on-update-map dissoc k)
                       kont)
                   ;; (update e :color lib/with-alpha
                   ;; (sin))
                   (assoc e
                     :color (q/lerp-color
                              (lib/->hsb base-color)
                              (lib/->hsb high-color)
                              (sin)))))])))


(defn ->vehicle-field
  [entities]
  (let [append-vehicles (into [] (map :id) entities)
        grid-width (long (Math/sqrt (count
                                     append-vehicles)))]
    (-> (lib/->entity :vehicle-field
                      {:particle-field (->vehicle-field-1
                                        grid-width)})
        (lib/live pe/particle-update)
        (lib/live
         [:vehicle-field
          (lib/every-n-seconds
           (fn [] (lib/normal-distr 1 1))
           (fn [e s k]
             (let [vehicle-activation
                   (pyutils/ensure-jvm
                    (p/read-activations
                     (:particle-field e)))]
               {:updated-state
                (reduce
                 (fn [s [id activation]]
                   #_(assoc-in s
                               [:eid->entity id :color]
                               (if (zero? activation)
                                 ((rand-nth [:black])
                                  defs/color-map)
                                 ((rand-nth [:cyan :hit-pink])
                                  defs/color-map)))
                   (if (zero? activation)
                     s
                     (->
                      s
                      (update-in
                       [:eid->entity id]
                       activation-flash
                       (:navajo-white defs/color-map)
                       (:green-yellow defs/color-map)
                       (fn [e]
                         (assoc e
                                :color (:navajo-white
                                        defs/color-map)))
                       ;; (if (zero? activation)
                       ;;   ((rand-nth [:black])
                       ;;   defs/color-map)
                       ;;   ((rand-nth [:cyan
                       ;;   :hit-pink])
                       ;;    defs/color-map))
                       )
                      )))
                 s
                 (map vector
                      append-vehicles
                      vehicle-activation))})))]))))

(defn +vehicle-field
  [state entities]
  (-> state
      (lib/append-ents [(->vehicle-field entities)])))




(defn ->yellow-heart
  ([] (->yellow-heart (lib/rand-on-canvas-gauss 0.2)))
  ([pos]
   (let [spawn-one
         (fn [e s _]
           (do
             (future (audio/play! (the-sound-of-birth)))
             {:updated-state
              (lib/append-ents
               s
               (vehicle-1
                {:pos (lib/position e)}))}))
         [e]
         (lib/->ray-source
          {:color
           (defs/color-map :navajo-white-tint)
           :on-double-click-map
           {:spawn-one spawn-one}
           :intensity 30
           :intensity-factor 1
           :kinetic-energy 1
           :no-stroke? true
           :on-collide-map {:burst (lib/cooldown 5 lib/burst)}
           :particle? true
           :pos pos
           :scale 1
           :z-index 5
           :ray-kind :yellow-heart})]
     (->
      e
      (lib/live
       [:spawns
        (lib/every-now-and-then 60 spawn-one)])
      (lib/live
       [:circular-shine-field
        (lib/every-n-seconds
         (fn [] (lib/normal-distr 0.5 0.5))
         (fn [ray s k]
           {:updated-state
            (lib/append-ents
             s
             [(let [e (lib/->circular-shine-1 ray)]
                (->
                 e
                 (assoc :no-stroke? true)
                 (assoc :color ((rand-nth
                                 [:hit-pink :white
                                  :navajo-white])
                                defs/color-map))
                 (assoc :on-update
                        [(lib/->grow
                          (* 2
                             (+ 1
                                (:intensity-factor
                                 ray
                                 0))))])
                 (assoc
                  :lifetime
                  (lib/normal-distr 2 1))))])}))])
      (lib/live [:intensity-osc update-intensity-osc])))))


(defn setup-internal
  [state]
  (-> state
      (lib/append-ents [(->yellow-heart)])
      (lib/live [:spawn-food
                 (lib/every-now-and-then
                  0.2
                  (fn [s _] (add-food-particle s)))])
      (assoc-in
       [:on-update-map :update-colors]
       (lib/every-n-seconds
        (let [last-temp (atom 0)]
          (fn [] (lib/normal-distr 0.1 0.1)))
        (let [black? (atom false)]
          (fn [s _]
            (let [color (defs/color-map
                          (if @black?
                            :black
                            (rand-nth [:hit-pink
                                       :heliotrope
                                       :green-yellow
                                       :white :cyan])))
                  _ (swap! black? not)]
              (lib/update-ents
               s
               (fn [ent]
                 (if-not (:update-colors1? ent)
                   ent
                   (if (not @black?)
                     (assoc ent
                            :color (defs/color-map :cyan))
                     (assoc ent
                            :color defs/black))))))))))))

(defmethod lib/setup-version :vehicle-1
  [state]
  (setup-internal state))

(defonce vehicle-feel (atom (hd/seed)))

(defonce feel->color-book
  (let [d (item-memory/codebook-item-memory-1
            (torch/stack (vec (hdd/clj->vsa*
                                (list
                                  ;; disgust
                                  :green-yellow
                                  ;; anger
                                  :heliotrope
                                    ;; neutral
                                    :navajo-white
                                  ;; 'fear'
                                  ;; (explore)
                                  :amethyst-smoke
                                    ;; joy
                                    :hit-pink)))))]
    (doseq [e (list :green-yellow
                    :heliotrope :navajo-white
                    :amethyst-smoke :hit-pink)]
      (item-memory/m-clj->vsa d e))
    d))

(comment
  (item-memory/m-cleanup* feel->color-book (hdd/clj->vsa* :hit-pink) 0.5))

(defn vehicle-feel->color [hv])

(defn vehicles
  [state]
  (let [entities
        (mapcat
         identity

         ;; (repeatedly 36 vehicle-1)
         ;; (repeatedly 50 vehicle-1)
         ;; (repeatedly 12 vehicle-1)

         (repeatedly 24 vehicle-1)
         ;; (repeatedly 50 vehicle-1)

         )]
    (-> state
        (lib/append-ents entities)
        ;; (+vehicle-field entities)
        )))

(defn temperature-bubble
  ([] (temperature-bubble (lib/rand-on-canvas-gauss 0.2)))
  ([pos]
   (let [e (let [d 25
                 hot-or-cold :hot
                 temp 1]
             (->
               (assoc
                (lib/->entity :circle) :transform
                (lib/->transform pos d d 5) :base-scale
                5 :no-stroke?
                true :color
                (:hit-pink defs/color-map)
                :temperature-bubble?
                true :kinetic-energy
                (lib/normal-distr 0.5 0.1) :hot-or-cold
                hot-or-cold :lifetime
                (lib/normal-distr 2 (Math/sqrt 2))
                :collides?
                true
                :on-collide-map
                {:flip-color
                 (lib/cooldown
                  (fn [] (lib/normal-distr 0.2 0.1))
                  (let [counter (atom 1)]
                    (fn [e other state k]
                      (if-not (:body? other)
                        e
                        (do (swap! counter inc)
                            (-> e
                                ;; (assoc-in
                                ;; [:transform
                                ;; :scale] (* 1.08
                                ;; @counter))
                                (assoc-in [:transform
                                           :base-scale]
                                          (* 1.08
                                             @counter))
                                (assoc
                                 :color
                                 (defs/color-map
                                   ([ ;; :deep-pink
                                     :black :cyan
                                     :hit-pink]
                                    (mod @counter
                                         2))))))))))}
                :d
                d :temp
                temp :z-index
                -10 :particle?
                true
                :draggable? false)
               (lib/live
                 [:scale-it
                  (let [sin (elib/sine-wave-machine 10
                                                    2000)]
                    (fn [e s k]
                      (let [v (sin)
                            base-scale (:base-scale e)]
                        (if (<= 0.9 v)
                          (assoc e :kill? true)
                          (-> e
                              (assoc-in [:transform :scale]
                                        (+ base-scale
                                           (sin)))))))
                    ;; (lib/live e
                    ;;           [:flash
                    ;;            (fn [e s k]
                    ;;              (if (lib/rang?
                    ;;              end-timer)
                    ;;                (-> e
                    ;;                    (update
                    ;;                    :on-update-map
                    ;;                    dissoc k)
                    ;;                    kont)
                    ;;                ;; (update e
                    ;;                :color
                    ;;                lib/with-alpha
                    ;;                ;; (sin))
                    ;;                (assoc e
                    ;;                       :color
                    ;;                       (q/lerp-color
                    ;;                               (lib/->hsb
                    ;;                               base-color)
                    ;;                               (lib/->hsb
                    ;;                               high-color)
                    ;;                               (sin)))))])
                  )])))]
     e)))


;; ----------------------------------------------------








(do
  (sketch
   {:background-color (:midnight-purple defs/color-map)
    :height nil
    :time-speed 2
    :v :vehicle-1
    :width nil})

  (swap! lib/event-queue (fnil conj []) vehicles)

  (do
    (defn add-overlay-text [state]
      (lib/append-ents state (overlay-text)))
    (swap! lib/event-queue (fnil conj []) add-overlay-text))

  (do
    (defn add-smell-history-overview [state]
      (lib/append-ents state (smell-history-overview)))
    (swap! lib/event-queue (fnil conj []) add-smell-history-overview)))

;; ----------------------------------------------------

(comment
  (filter :sensor?  (lib/entities @lib/the-state))
  )


(comment
  (reset! lib/the-state {})
  (reset! lib/event-queue [])
  (filter :spawner? (lib/entities @lib/the-state))


  (:smell-history
   (let [{:keys [id]} (:selection @lib/the-state)]
     ((lib/entities-by-id @lib/the-state) id)))


  (key
   (first
    (sort-by
     val
     (merge
      (update-vals ant-types (constantly 0))
      (frequencies
       [:worker :worker :gatherer :soldier :worker :gatherer
        :gatherer :worker :gatherer :worker])))))
  :caretaker

  ([:caretaker 0] [:soldier 1] [:gatherer 4] [:worker 5])


  (future (do (audio/play! (audio/->audio {:duration 0.2
                                           :frequency 250}))
              (audio/play! (audio/->audio {:duration 0.2
                                           :frequency 500}))
              (audio/play! (audio/->audio {:duration 0.2
                                           :frequency 300}))
              (audio/play! (audio/->audio {:duration 0.2

                                           :frequency 400}))))



  (select-keys
   (let [{:keys [id]} (:selection @lib/the-state)]
     ((lib/entities-by-id @lib/the-state) id))
   [:ant-data :smell-history])


  {:ant-data {:audio-freq 250
              :color {:h 36 :s 32 :v 100}
              :exploration-drive 0.2
              :explore-around-yellow-heart 0
              :hunger-for-green-food 0
              :kind :caretaker
              :love-for-yellow-heart 10}
   :smell-history [:gatherer :soldier :soldier]}





  (key
   (first
    (shuffle
     (first
      (partition-by
       val
       (sort-by
        val
        {:foo 0 :bar 0 :xar 20}))))))


  (future
    (audio/play!
     (audio/->audio
      {:duration 0.2
       :frequency 160}))
    (audio/play!
     (audio/->audio
      {:duration 0.2
       :frequency 150}))))
