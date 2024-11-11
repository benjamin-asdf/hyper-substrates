(ns vehicles.1
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
    [animismic.lib.blerp :as b]
    [animismic.lib.particles-core :as pe]
    [ftlm.vehicles.cart :as cart]
    [animismic.lib.vehicles :as v]
    [libpython-clj2.require :refer [require-python]]
    [libpython-clj2.python :refer [py. py..] :as py]
    [ftlm.vehicles.art.physics :as phy]
    [ftlm.disco.middlewares.picture-jitter :as
     picture-jitter]))

(defn draw-state
  [state]
  (q/color-mode :hsb)
  (q/background (lib/->hsb (-> state
                               :controls
                               :background-color)))
  (do (q/stroke-weight 0)
      (lib/draw-entities state))
  ;; (q/color-mode :rgb)
  ;; (q/fill 0 0 0 10)
  ;; (q/rect 0 0 (* 2 2560) (* 2 1920))
  )

(def glyph-size 18)
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
        ;; picture-jitter (:picture-jitter state)
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
                      lib/update-late-update-map
                      lib/transduce-signals
                      ;; those 2 are heavy,
                      lib/track-components
                      lib/track-conn-lines
                      ;; also heavy:
                      lib/update-collisions
                      ;;
                      phy/physics-update-2d
                      ;; -----------------
                    ))]
    (merge state new-state)))

(defn update-state
  [state]
  (let [current-tick (q/millis)
        dt (* (:time-speed (lib/controls))
              (/ (- current-tick
                    (:last-tick @lib/the-state 0))
                 1000.0))]
    (lib/update-timers dt)
    (merge state
           (swap! lib/the-state update-state-inner
                  dt
                  current-tick))))

(defn setup
  [controls]
  (q/frame-rate 60)
  (q/rect-mode :center)
  (q/color-mode :hsb)
  (q/text-size glyph-size)
  (q/text-font (q/create-font "Fira Code Bold" glyph-size)
               glyph-size)
  (let [state {:controls controls :on-update []}
        state (-> state
                  lib/setup-version)]
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
    [1500 1500]
    ;; [500 500]
    :setup (partial setup controls)
    :update #'update-state
    :draw #'draw-state
    ;; :features [:keep-on-top]
    :middleware [m/fun-mode m/navigation-2d
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

(defn ->ray-source
  ([] (->ray-source (lib/rand-on-canvas-gauss 0.2)))
  ([pos]
   (let [[e] (lib/->ray-source
              {:color (:mint defs/color-map)
               :intensity 30
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
               :scale 0.75
               :shinyness nil})]
     (-> e
         (lib/live
          [:circular-shine-radio
           (lib/every-n-seconds
            (fn [] (lib/normal-distr (/ 1.5 3) (/ 1.5 3)))
            (fn [ray s k]
              {:updated-state
               (lib/append-ents
                s
                [(let [e (lib/->circular-shine-1 ray)]
                   (-> e
                       (assoc :color
                              (lib/with-alpha
                                (:yellow
                                 defs/color-map)
                                0))
                       (assoc :stroke-weight 3)
                       (assoc :stroke (:color ray))
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
                               3
                               (Math/sqrt
                                3)))))])}))])))))

;; ------------------------------------------------------------

(defn vehicle-1
  []
  (let [cart
        (cart/->cart
         {:body {:color (:navajo-white defs/color-map)
                 :kinetic-energy 1
                 :particle? true
                 :scale 1
                 :stroke-weight 1
                 :transform
                 (assoc (lib/->transform
                         (lib/rand-on-canvas-gauss
                          0.5)
                         30
                         60
                         1)
                        :rotation (q/random q/TWO-PI))}
          :components
          [[:cart/motor :ma
            {:anchor :bottom-middle
             :corner-r 5
             :hidden? false
             :on-update [(lib/->cap-activation)]
             :rotational-power 0.02}]
           ;; ----------------
           ;; Temperature sensor
           [:cart/sensor :hot-temperature-sensor
            {:anchor :top-middle
             :hot-or-cold :hot
             :modality :temperature}]
           ;; ----------------------------
           [:brain/connection :_
            {:decussates? false
             :destination [:ref :ma]
             :f :excite
             :hidden? true
             :source [:ref
                      :hot-temperature-sensor]}]]})]
    cart))

(defmethod lib/setup-version :vehicle-1 [state]
  state)

(defn vehicles
  [state]
  (let [entities (mapcat identity
                   (repeatedly 12 vehicle-1))]
    (-> state
        (lib/append-ents entities))))

(defn temperature-bubble
  ([] (temperature-bubble (lib/rand-on-canvas-gauss 0.2)))
  ([pos]
   (let [e (let [d 25
                 hot-or-cold :hot
                 temp 1]
             (-> (assoc (lib/->entity :circle)
                        :transform (lib/->transform pos d d 5)
                        :base-scale 5
                        :clickable? true
                        :draggable? true
                        :no-stroke? true
                        :color (:hit-pink defs/color-map)
                        :temperature-bubble? true
                        :kinetic-energy (lib/normal-distr 0.5 0.1)
                        :hot-or-cold hot-or-cold
                        :d d
                        :temp temp
                        :z-index -10
                        :particle? true)))]
     e)))

;; ----------------------------------------------------

(do (sketch {:background-color 0
             :height nil
             :time-speed 3.5
             :v :vehicle-1
             :width nil})
    (swap! lib/event-queue (fnil conj []) vehicles)
    ;; (swap! lib/event-queue (fnil conj [])
    ;;   (fn [s]
    ;;     (lib/append-ents s
    ;;                      (repeatedly 5
    ;;                                  temperature-bubble))))
    )



;; ----------------------------------------------------
