;; extended lib.
;; file was getting too big
(ns ftlm.vehicles.art.extended
  (:require
   [quil.core :as q :include-macros true]
   [ftlm.vehicles.art.lib :as lib]
   [ftlm.vehicles.art.defs :as defs]
   [ftlm.vehicles.art.defs]))

(defn ->brownian-lump
  [{:keys [spread particle-size pos togethernes-threshold
           count colors]
    :or {colors [[0 255 255]]
         count 10
         particle-size 10
         spread 8
         togethernes-threshold (or spread (* 2 spread))}}]
  (let [lump (assoc (lib/->entity :lump)
                    :hidden? true
                    :lump? true
                    :position pos
                    :spread spread)]
    (into [lump]
          (map
           (fn []
             (let [spawn-pos
                   [(lib/normal-distr (first pos) spread)
                    (lib/normal-distr (second pos) spread)]]
               (->
                (merge
                 (lib/->entity :circle)
                 {:color (rand-nth colors)
                  :draggable? false
                  :kinetic-energy 0.2
                  :on-update
                  [(fn [e]
                     (let [threshold
                           togethernes-threshold
                           dist (lib/distance (lib/position e)
                                              pos)]
                       (if (< threshold dist)
                         (assoc (lib/orient-towards e pos)
                                :acceleration 2
                                :angular-acceleration 0)
                         e)))]
                  :particle? true
                  :transform
                  (assoc (lib/->transform spawn-pos
                                          particle-size
                                          particle-size
                                          1)
                         :rotation (lib/angle-between spawn-pos
                                                      pos))
                  :z-index 10})))))
          (range count))))

(defn ->oxygen
  [opts]
  (lib/flatten-components
    [(merge
       (lib/->odor-source
         (merge opts {:fragrances #{:oxygen}} (:odor opts)))
       {:components (->brownian-lump
                      (assoc opts
                        :colors
                          (into []
                                (repeatedly
                                  4
                                  (fn []
                                    {:h 178
                                     :s (lib/normal-distr 20 10)
                                     :v 255})))
                        :spread 20
                        :count 15
                        :particle-size 8
                        :togethernes-threshold 50))
        :draggable? false
        :oxygen? true})]))

(defn ->organic-matter
  [opts]
  (lib/flatten-components
    [(merge (lib/->odor-source
              (merge opts {:fragrances #{:organic-matter}} (:odor opts)))
            {:components (->brownian-lump opts)
             :draggable? false
             :food? true
             :organic-matter? true})]))

(defn ->temperature-bubble-1
  [{:as opts
    :keys [pos d temp max-temp low-color high-color
           hot-or-cold]}]
  [(merge
    (assoc
     (lib/->entity :circle)
     :transform (lib/->transform pos d d 1)
     :no-stroke? true
     :color
     (q/lerp-color
      (lib/->hsb low-color)
      (lib/->hsb high-color)
      (lib/normalize-value-1 0 max-temp temp))
     :temperature-bubble? true
     :hot-or-cold hot-or-cold
     :d d
     :temp temp
     :z-index -10
     :particle? true
     :draggable? true)
    opts)])

(defn ->temperature-bubble [opts]
  (fn [opts-1]
    (->temperature-bubble-1 (merge opts opts-1))))

(defn ->breath
  [initial-scale size speed]
  (let [mystate (atom {:speed speed :time 0})]
    {[:breath :play-with-speed]
     (lib/every-n-seconds
      speed
      (fn [_ _ _]
        (swap! mystate update
               :speed
               (constantly (lib/normal-distr speed (/ speed 2))))
        nil))
     [:breath :rotate]
     (lib/every-n-seconds
      speed
      (fn [e _ _]
        (update e
                :angular-acceleration
                +
                (* (/ speed 3)
                   (lib/normal-distr 0
                                     (/ (mod (:time @mystate)
                                             q/TWO-PI)))))))
     [:breath :scale]
     (fn [e _ _]
       (-> e
           (update-in
            [:transform :scale]
            (fn [_scale]
              (let [progress (/ (:time @mystate) 1)]
                (q/lerp
                 initial-scale
                 (* initial-scale size)
                 (+ 1 (q/sin (* q/PI progress)))))))))
     [:breath :time]
     (fn [_ _ _]
       (let [t (fn [{:as s :keys [speed]}]
                 (-> s
                     (update :time + (* speed lib/*dt*))))]
         (swap! mystate t))
       nil)}))

(defn ->plasma-balls
  [{:keys [from to color start-entity]
    :or {color {:a 0.8 :h 0 :s 100 :v 100}}}]
  (map-indexed
   (fn [_ _]
     (let [pos from]
       (->
        (merge
         (lib/->entity :circle)
         {:acceleration 150
          :color color
          ;; :kinetic-energy 0.1
          :lifetime (lib/normal-distr 5 5)
          :on-update-map
          {:kill (fn [e _ _]
                   (if (<= (lib/distance (lib/position e) to)
                           10)
                     (assoc e :lifetime 0)
                     e))
           :target (lib/every-n-seconds
                    1.5
                    (fn [e _ _]
                      (let [mag (lib/distance (lib/position e)
                                          to)]
                        (-> (lib/orient-towards e to)
                            (update :acceleration
                                    +
                                    (lib/normal-distr
                                     (* mag 3)
                                     (* mag 2)))))))}
          :particle? true
          :transform
          (lib/->transform (lib/position start-entity) 10 10 1)
          :z-index -1})
        (lib/orient-towards pos))))
   (range 1)))

(defn ->color-back-and-forth-zagged
  [duration high low]
  (let [s (atom {:time-since 0})]
    (fn [e _ _]
      (swap! s update :time-since + lib/*dt*)
      (let [progress (lib/normalize-value-1 0 duration (mod (:time-since @s) duration))]
        (assoc e :color (q/lerp-color high low progress))))))


(defn ->activation-burst
  [state-atom id]
  (fn [_ _ _]
    (let [s @state-atom
          e ((lib/entities-by-id s) id)]
      (when e
        (swap! state-atom update-in [:eid->entity id :activation] + 100)))
    nil))

(defn with-electrode-sensitivity
  [e]
  (assoc-in e
    [:on-late-update-map :electrode-sensitivity]
    (fn [e _s _k]
      (if-let [electrode-input (:electrode-input e)]
        (->
         e
         (dissoc :electrode-input)
         (update :activation + electrode-input))
        e))))

(defn assembling-multi-line
  [{:keys [source dest]}]
  (fn [e s _]
    (let [source-e ((lib/entities-by-id s) source)
          dest-e ((lib/entities-by-id s) dest)
          start-pos (lib/position source-e)
          end-pos (lib/position dest-e)
          update-pos
          (fn [e]
            (let [start-pos (update start-pos 1 (fn [v] (+ v (rand-nth (range -10 10 5)))))
                  end-pos (update end-pos 0 (fn [v] (+ v (rand-nth (range -10 10 5)))))]
              (assoc e
                     :vertices [start-pos [(first end-pos) (second start-pos)] end-pos])))
          e (assoc e
                   :hidden? (some :dragged? [source-e dest-e]))
          e (cond-> e
              (or (not= start-pos (get-in e [:vertices 0]))
                  (not= end-pos (get-in e [:vertices 2])))
              update-pos)]
      e)))

(defn rect-multi-line-vertices
  [source-e dest-e]
  (let [start-pos (lib/position source-e)
        end-pos (lib/position dest-e)
        start-pos (update
                    start-pos
                    1
                    (fn [v]
                      (+ v (rand-nth (range -20 20 5)))))
        end-pos (update end-pos
                        0
                        (fn [v]
                          (+ v
                             (rand-nth (range -20 20 5)))))]
    [start-pos [(first end-pos) (second start-pos)]
     end-pos]))

(defn rect-line-vertices-1
  [start-pos end-pos]
  (let [start-pos (update
                   start-pos
                   1
                   (fn [v]
                     (+ v (rand-nth (range -10 10 5)))))
        end-pos (update end-pos
                        0
                        (fn [v]
                          (+ v
                             (rand-nth (range -10 10 5)))))]
    [start-pos [(first end-pos) (second start-pos)]
     end-pos]))

(defn rect-line-vertices-2
  [start-pos end-pos low high]
  (let [start-pos (update
                   start-pos
                   1
                   (fn [v]
                     (+ v (rand-nth (range low high 1)))))
        end-pos (update end-pos
                        0
                        (fn [v]
                          (+ v
                             (rand-nth (range low high 1)))))]
    [start-pos [(first end-pos) (second start-pos)]
     end-pos]))


(defn ->fade
  ([speed] (->fade speed :color))
  ([speed k]
   (fn [e s _]
     (update e
             k
             (fn [c]
               (let [c (lib/->hsb c)]
                 (lib/with-alpha c
                   (* (- 1 (* lib/*dt* speed))
                      (q/alpha c)))))))))

(defn from-right [amount]
  (- (q/width) amount))

(defn from-bottom [amount]
  (- (q/height) amount))

(defn flash
  [e]
  (-> e
      (assoc :lifetime 3)
      (lib/live [:fade (->fade 1)])))

(defn ->flash-of-line
  ([pos-1 pos-2]
   (->flash-of-line pos-1 pos-2 {:color (:cyan defs/color-map)}))
  ([pos-1 pos-2 opts]
   (lib/->entity
    :multi-line
    (merge
     {:lifetime 3
      :on-update-map {:fade (->fade 1)}
      :stroke-weight 1
      :transform (lib/->transform pos-1 1 1 1)
      :vertices (rect-line-vertices-1 pos-1 pos-2)
      :z-index -2}
     opts))))



;; (defn grid [draw-i]
;;   (lib/->entity
;;    :nn-area
;;    (merge
;;     {:color (:cyan defs/color-map)
;;      :draw-functions
;;      {:1 (fn [e]
;;            (let [neurons (ac/read-activations (:ac-area
;;                                                e))
;;                  i->pos (fn [i] ((e :i->pos) e i))]
;;              (q/with-stroke
;;                nil
;;                (doall
;;                 (for [i neurons :let [pos (i->pos i)]]
;;                   (q/with-translation pos (draw-i i)))))))}
;;      :i->pos (fn [{:keys [transform]} i]
;;                (let [[x y] (:pos transform)
;;                      coll (mod i grid-width)
;;                      row (quot i grid-width)
;;                      x (+ x (* coll spacing))
;;                      y (+ y (* row spacing))]
;;                  [x y]))
;;      :next-color (constantly (:cyan defs/color-map))
;;      :spacing spacing}
;;     opts)))


(defn grid-pos-1
  [spacing grid-width [x y] i]
  (let [coll (mod i grid-width)
        row (quot i grid-width)
        x (+ x (* coll spacing))
        y (+ y (* row spacing))]
    [x y]))

(defn grid-pos [{:keys [transform grid-width spacing]} i]
  (let [[x y] (:pos transform)]
    (grid-pos-1 spacing grid-width [x y] i)))

(defn ->tiny-breath
  [{:keys [start stop speed]}]
  (let [t (atom (rand q/TWO-PI))]
    (fn [e _ _]
      (swap! t + lib/*dt*)
      (update-in
        e
        [:transform :scale]
        (fn [_]
          (q/lerp start stop (q/sin (* speed @t))))))))


(defn ->flash-line-tracking
  [e end-pos]
  (->
    (assoc (->flash-of-line (lib/position e) end-pos) :color
           (:color e))
    (assoc-in
      [:on-update-map :find-pos]
      (fn [line-e s _]
        (let [start-pos (lib/position
                          ((lib/entities-by-id s) (:id e)))]
          (update-in line-e
                     [:vertices]
                     (constantly (rect-line-vertices-1
                                   start-pos
                                   end-pos))))))))

(defn ->clock-flower
  [{:as opts :keys [pos radius count draw-element]}]
  (let [angle-step (/ 360 count)]
    (lib/->entity
      :clock-circles
      (merge
        opts
        {:draw-functions
           {:1 (fn [e]
                 (doall
                   (map-indexed
                     (fn [idx _]
                       (let [angle (* idx angle-step)
                             center-pos pos
                             sub-pos (lib/position-on-circle
                                       center-pos
                                       radius
                                       angle)]
                         (if draw-element
                           (q/with-translation
                               ;; sub-pos
                             [(first sub-pos) (second sub-pos)]

                               (draw-element e idx))
                           (do
                             (q/stroke-weight 2)
                             (q/with-stroke
                               (lib/->hsb defs/white)
                               (q/with-fill
                                 (or (when (:i->fill e)
                                       ((:i->fill e) e idx))
                                     (lib/with-alpha
                                       (lib/->hsb
                                         defs/white)
                                       0))
                                 (q/ellipse (first sub-pos)
                                            (second sub-pos)
                                            20
                                            20)))))))
                     (range count))))}
         :transform (lib/->transform pos 100 100 1)}))))

(defn sine-wave-update
  [x speed cycle-duration]
  (let [fade-factor (-> (* (/ (q/millis) cycle-duration)
                           q/TWO-PI)
                        (Math/sin)
                        (Math/abs))
        wave-value (* fade-factor (+ x (* lib/*dt* speed)))]
    wave-value))

(defn sine-wave-machine
  [speed cycle-duration]
  (let [state (atom 0)]
    (fn
      ([deref?] @state)
      ([]
       (swap! state sine-wave-update
         speed
         cycle-duration)))))

(defn ->text
  [opts]
  (lib/->entity :text
                (merge
                  {:draw-functions
                     {:f (fn [e]
                           (def e e)
                           (let [transform (:transform e)
                                 [x y] (:pos transform)
                                 {:keys [width height scale
                                         rotation]}
                                   transform]
                             ;; (q/color-mode :rgb)
                             ;; (q/color 0 255 0)
                             (q/with-translation
                               [x y]
                               (q/with-rotation
                                 [rotation]
                                 (q/text (:text e) 0 0)))))}
                   :text "+"}
                  opts)))

(defn in-bounds? [index coll]
  (and (>= index 0)
       (< index (count coll))))

(defn text-raindrop
  [opts]
  (lib/->entity
    :text-raindrop
    (merge
      {:draw-functions
         {:f (fn [e]
               (let [transform (:transform e)
                     [x y] (:pos transform)
                     {:keys [width height scale rotation]}
                       transform
                     c (if (in-bounds? (:index e) (:text e))
                         (nth (:text e) (:index e))
                         (rand-nth ["#" "?" "!" "@"]))]
                 (println c (:index e))
                 (q/with-translation
                   [x (+ y (* (:glyph-size e) (:index e)))]
                   (q/with-rotation
                     [rotation]
                     (q/text (str c) 0 0)))))}
       :glyph-size 18
       :text "+"}
      opts)))

(defn digital-raindrop
  [opts]
  (lib/live
    (text-raindrop
      (merge
        {:color (:ice-cyan defs/color-map)
         :index 0
         ;; :loop? true
         :text (map char
                 (repeatedly
                   (:length opts
                            (q/floor (/ (q/height) 18)))
                   #(rand-nth (concat (range 32 128)
                                      (range 160 383)
                                      (range 880 1024)
                                      (range 8592 8600)
                                      (range 8704 8720)))))
         :transform (lib/->transform (lib/rand-on-canvas)
                                     10 10
                                     1 q/PI)}
        opts))
    (lib/every-n-seconds
     (:every-when opts 1)
     (fn [e s k]
       (if-not (in-bounds? (:index e)
                           (:text e))
         (if (:loop? e)
           (assoc e :index 0)
           (assoc e :kill? true))
         (update e :index inc))))))
