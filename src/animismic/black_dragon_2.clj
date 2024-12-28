(ns animismic.black-dragon-2
  (:require
   [emmy.env :as e]
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
    picture-jitter]
   [ftlm.vehicles.art.event-q-middleware :as evtq]
   [ftlm.disco.middlewares.time-warp :as time-warp]
   [ftlm.disco.wave :as wave]))

(def glyph-size 18)

(defn draw-state
  [state]
  ;; (q/color-mode (rand-nth [:rgb :hsb]))
  ;; (q/color-mode :rgb)
  (q/color-mode :hsb)
  (q/fill
   (lib/with-alpha

     (lib/->hsb (:background-color state))

     0.2))
  (q/rect 0 0 (* 2 2560) (* 2 1920))
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

(defn update-state
  [state]
  (let [env (lib/env state)
        current-tick (q/millis)
        ;; state (update state
        ;;               :controls
        ;;               merge
        ;;               (user-controls/controls))
        dt (* (:time-speed (lib/controls))
              (/ (- current-tick (:last-tick state 0))
                 1000))]
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
        lib/update-rhythm-timeline
        ;;
        phy/physics-update-2d
        lib/update-timers-v2))))


(defn setup
  [opts]
  (q/frame-rate 60)
  (q/rect-mode :center)
  (q/color-mode :hsb)
  (q/text-size glyph-size)
  (q/text-font (q/create-font "Fira Code Bold" glyph-size)
               glyph-size)
  (let [state {:background-color (:background-color opts)
               :controls opts
               :foo :bar
               :on-update []}
        state (-> state
                  lib/setup-version)]
    state))

(defn sketch
  [{:as opts
    :keys [width height sketch-id]
    :or {height 800 width 1000}}]
  (q/sketch
   :events-q {:sketch-id sketch-id}
   :size
   ;; hard coding my monitor, else it was going to
   ;; another monitor
   [1920 1080]
   ;; [500 500]
   :setup
   (comp (:setup opts identity)
         (partial setup opts))
   :update #'update-state
   :draw #'draw-state
   :middleware [m/fun-mode m/navigation-2d
                picture-jitter/picture-jitter
                evtq/events-middleware time-warp/time-warp]
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


(def tile->arc
  {:ne [q/PI (+ q/PI q/HALF-PI)]
   :nw [(+ q/PI q/HALF-PI) q/TWO-PI]
   :se [q/HALF-PI q/PI]
   :sw [0 q/HALF-PI]})

(def arc-position
  [:ne :nw :se :sw])

(defn worm-growth [])



;;
;; Dragon grid:
;; ---------
;;
;;
;;
;;   0    1  , ...    ~ width / 2 * segment-length
;;
;; +----+----+
;; |    |    |  0
;; +----X----+
;; |    |    |  1
;; +----+----+
;;              ,..
;;
;;                  ~ height / 2 * segment-length
;;
;;
;; 4 elements in the dragon grid make one anulus
;;
;; values:
;; 1 - on,
;; 0 - off
;;
;;
;;

(defn dragon-grid
  [shape]
  #_(def shape [2 2])

  (->
   (torch/zeros shape)
   (py/set-item! [0 0] 1)
   (py/set-item! [1 1] 1)))

(comment
  (dragon-grid [4 4]))

(defn dragon-grid->arcs
  [grid]
  (let [grid (torch/reshape
               (py.. grid (unfold 0 2 2) (unfold 1 2 2))
               [-1 4])
        arcs (->> grid
                  (map (fn [elms]
                         (apply concat
                           (map (fn [el dir]
                                  (when-not (zero? (py..
                                                     el
                                                     item))
                                    [dir]))
                             elms
                             arc-position)))))]
    arcs))

(comment
  (dragon-grid->arcs
   (dragon-grid [4 4]))
  (dragon-grid->arcs
   grid))



(comment

  (mapcat identity
          (for [row (into []
                          (dragon-grid->patt (dragon-grid [4 4])))]
            (map-indexed (fn [idx v]
                           (when-not (zero? (py.. v item))
                             [(arc-position idx)]))
                         row)))
  )



(defn black-dragon
  [{:keys [pos width height]}]
  (let [->grid
        (fn []
          (let [positions (for [x (range 0 width 80)]
                            (for [y (range 0 height 80)]
                              [x y]))
                shape [(* 2 (count positions))
                       (* 2 (count (first positions)))]]
            {:grid (dragon-grid shape)
             :positions positions}))]
    (->
     (lib/->entity
      :bd
      {:draw-functions
       {:f (fn [e]
             (def e e)
             (q/with-rotation
                 [0
                  ;; (lib/rotation e)
                  ]
                 (q/with-translation
                     (lib/position e)
                     (doall
                      (for [[pos arcs]
                            (map
                             vector
                             (apply concat (-> e :grid :positions))
                             (dragon-grid->arcs (:grid (:grid e))))]
                        (q/with-translation
                            pos
                            (do
                              (q/no-fill)
                              (q/stroke-weight 10)
                              (doall
                               ;; (def arcs arcs)
                               (for [
                                     ;; {:keys [color arc]}
                                     arc
                                     (map tile->arc arcs)]
                                 (do
                                   (q/stroke
                                    (lib/->hsb 0
                                               ;; color
                                               ))
                                   (q/arc 0
                                          0
                                          80
                                          80
                                          (first arc)
                                          (second arc)))))
                              (q/no-stroke)
                              (q/with-fill
                                  (lib/->hsb (:color e 0))
                                  (q/ellipse 0 0
                                             30 30)))))))))}
       :grid (->grid)
       :kinetic-energy 0.2
       :particle? true
       ;; :patt (->patt)
       :transform (lib/->transform pos 0 0 1 0)})
     #_(lib/live (lib/every-now-and-then
                0.025
                (fn [e s k] (assoc e :patt (->patt)))))
     #_(lib/live (lib/every-now-and-then
                  0.2
                  (fn [e s k]
                    (assoc e
                           :color (rand-nth [ ;; :red
                                             :white :red
                                             ;; :white
                                             ;; :green-yellow
                                             ;; :black
                                             ]))))))))













(defn add-black-dragon [state]
  (lib/append-ents
   state
   [ ;; (black-dragon {:pos [200 200]
    ;;                :width 1600
    ;;                :height 900})
    (black-dragon {:pos [100 100]
                   :width 1600
                   :height 900})
    ;; (black-dragon {:pos [0 0]
    ;;                :width 800
    ;;                :height 800})
    ]))




(sketch
 {:background-color :white
  :height nil
  :setup (comp #_(fn [state]
                   (lib/live state
                             (lib/every-now-and-then
                              0.5
                              (fn [s k]
                                (assoc s
                                       :background-color
                                       (rand-nth
                                        [ ;; :red
                                         ;; :green-yellow
                                         :black :black
                                         :black
                                         :white]))))))
               add-black-dragon)
  :sketch-id :s
  :time-speed 1
  :v :black-dragon
  :width nil})
