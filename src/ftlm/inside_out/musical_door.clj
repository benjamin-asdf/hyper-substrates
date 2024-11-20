(ns ftlm.inside-out.musical-door
  (:require
    [animismic.lib.particles :as p]
    [fastmath.core :as fm]
    [clojure.java.io :as io]
    [clojure.string :as str]
    [clojure.data.json :as json]
    [quil.middleware :as m]
    [ftlm.vehicles.art.lib :refer [*dt*] :as lib]
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
    [animismic.lib.blerp :as b]))

;; or 'alphabet'
(def colors-of-mind
  [:orange :cyan :very-blue :heliotrope :yellow
   :green-yellow :red])

(defn env [_state] {})

(defn draw-state
  [state]
  (q/background (lib/->hsb (-> state :controls :background-color)))
  (q/stroke-weight 1)
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
  [state]
  ;; state
  ;; (when (< 10 (:t state 0))
  ;;   (q/exit))
  (let [current-tick (q/millis)
        ;; state (update state
        ;;               :controls
        ;;               merge
        ;;               (user-controls/controls))
        dt (* (:time-speed (lib/controls))
              (/ (- current-tick (:last-tick state 0))
                 1000.0))
        state (binding [*dt* dt]
                (-> state
                    (update :t (fnil inc 0))
                    (assoc :last-tick current-tick)
                    lib/apply-update-events
                    lib/update-update-functions
                    lib/update-state-update-functions
                    lib/apply-events
                    ;; (lib/apply-events (:event-q
                    ;; state))
                    (lib/update-ents
                     #(update-entity % state (env state)))
                    lib/update-late-update-map
                    lib/transduce-signals
                    lib/track-components
                    lib/track-conn-lines
                    lib/ray-source-collision-burst
                    lib/kill-entities))]
    state))

(defn update-state
  [_]
  (let [state @lib/the-state
        state (update-state-inner state)]
    (reset! lib/the-state state)
    state))

(defn setup
  [controls]
  (q/frame-rate 60)
  (q/rect-mode :center)
  (q/color-mode :hsb)
  (q/background (lib/->hsb (-> controls
                               :background-color)))
  (let [state {:controls controls :on-update []}
        state (-> state
                  lib/setup-version)]
    (reset! lib/the-state state)))



(defn sketch
  [{:as controls
    :keys [width height]
    :or {height 800 width 1000}}]
  (q/sketch
   :size [width height]
   :setup (partial setup controls)
   :update update-state
   :draw draw-state
   :features [:keep-on-top]
   :middleware [m/fun-mode]
   :title "hyper-substrates"
   :key-released (fn [state event] state)
   :mouse-pressed (comp #(reset! lib/the-state %)
                        lib/mouse-pressed)
   :mouse-released (comp #(reset! lib/the-state %)
                         lib/mouse-released)
   :mouse-wheel (comp #(reset! lib/the-state %)
                      lib/mouse-wheel)
   :on-close (reset! lib/the-state nil)))

(defn draw-door-element
  [{:keys [color strength visible? blink-state]}]
  (when
      visible?
      (q/with-stroke nil
        (q/with-fill
            (some-> color
                    (defs/color-map)
                    (lib/->hsb)
                    ;; (q/lerp-color
                    ;;  (lib/->hsb defs/white)
                    ;;  (* 0.5 strength))
                    )
            (when strength
              (let [size (* 20 (+ 1 strength))]
                (q/ellipse 0 0 size size)))))))

(defn blink
  [{:as s :keys [elements]} _ _]
  (def elements elements)
  (assoc s
    :elements
      (into []
            (map
              (fn [{:as e :keys [blinks?]}]
                (cond-> e
                  blinks?
                    (update
                      :strength
                      (let [speed 2
                            cycle-duration 1000]
                        (fn [alpha]
                          (let [fade-factor
                                  (-> (* (/ (- (q/millis)
                                               (:start e))
                                            cycle-duration)
                                         q/TWO-PI)
                                      (Math/sin)
                                      (Math/abs))
                                wave-value (* fade-factor
                                              (+ alpha
                                                 (*
                                                   lib/*dt*
                                                   speed)))]
                            wave-value))))
                  (some-> (:end e)
                          (< (q/millis)))
                    (assoc :blinks?
                      false :visible?
                      false))))
            elements)))

(defn door
  []
  (->
    (elib/->clock-flower
      {:count 12
       :draw-element (comp draw-door-element
                           (fn [e idx]
                             (get (:elements e) idx)))
       :elements (into []
                       (map-indexed (fn [idx color]
                                      {:color color
                                       :idx idx
                                       :strength (rand)
                                       :visible? false})
                                    colors-of-mind))
       :pos [200 200]
       :radius 100})
    (lib/live [:blink blink])
    (lib/live
      (let [next-blink
              (atom (cycle (concat [0]
                                   (range
                                     (count
                                       colors-of-mind)))))]
        ;; [:blink-start
        ;;  (lib/every-n-seconds
        ;;   0.5
        ;;   (fn [e s _]
        ;;     (let [next-idx (first (swap! next-blink
        ;;                                  next))]
        ;;       (update-in e
        ;;                  [:elements next-idx]
        ;;                  merge
        ;;                  {:blinks? true
        ;;                   :end (+ (q/millis) 1000)
        ;;                   :start (q/millis)
        ;;                   :visible? true}))))]
        [:blink-start
         (lib/every-n-seconds
           0.5
           (fn [e s _]
             (if-let [next-idx
                        (first
                          (shuffle
                            (clojure.set/difference
                              (into #{}
                                    (range
                                      (count
                                        colors-of-mind)))
                              (into
                                #{}
                                (comp
                                  (filter
                                    (some-fn
                                      (fn [e]
                                        (when-let [end (:end
                                                         e)]
                                          (< (- (q/millis)
                                                end)
                                             1000)))
                                      :blinks?))
                                  (map :idx))
                                (:elements e)))))]
               (update-in e
                          [:elements next-idx]
                          merge
                          {:blinks? true
                           :end (+ (q/millis) 1000)
                           :start (q/millis)
                           :visible? true})
               e)))]))))

(defmethod lib/setup-version :musical-door-1
  [state]
  (-> state
      (lib/append-ents
       [(door)])))

(sketch {:background-color 0
         :height 600
         :time-speed 3
         :v :musical-door-1
         :width 800})




;; ------------------------------------
;; inside out:
;;
;; - door
;; - memory orbs
;; - control panel
;; - imagination booths
;; - the self model
;; - emotions as tiny control circuits (Jaak Panksepp)
;;
